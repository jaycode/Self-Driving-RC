# To use, run:
# `sudo su`
# `python test.py --model [model-path] --dir [test-dir-path] [-d] [-t]`

# === Model Path ===
# `model-path` is the path to model definition json. Model weights should be
# contained in the same path.


# === Training Dir ===
# Set `dir` to a directory that contains the following:
# - A directory with images to see how the car drives with these data.
# - A csv file for ground truth.

# === Drive ===
# When flag `-d` is included, send command to the actuators.

# === Throttle ===
# By default, this script does not actuate throttle. To allow it to
# send throttle commands, include flag `-t`.

import numpy as np
import cv2
from datetime import datetime
import os
import argparse
import h5py
from keras.models import load_model
import time
import re
import glob
import csv
import math
import matplotlib.pyplot as plt
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
# Path to Computer root directory
ROOT_DIR = os.path.realpath(os.path.join(dir_path, '..'))

sys.path.append(ROOT_DIR)
from libraries.helpers import choose_port

# This is the smallest current camera may support.
# (i.e. setting CAP_PROP_FRAME_WIDTH and HEIGHT smaller than this won't help)
TARGET_WIDTH = 320
TARGET_HEIGHT = 240
TARGET_CROP = ((60, 20), (0, 0))

# TODO: This is currently throttle value but we will update it once we got
#       accelerometer.
set_speed = 130

MIN_THROTTLE = 50
MAX_THROTTLE = 130
THROTTLE_P=0.3
THROTTLE_I=0.08

HOST_AUTO_STEER = b's'
HOST_AUTO_THROTTLE = b't'

# Try out several ports to find where the microcontroller is.
ports = ["/dev/ttyUSB0", "/dev/ttyUSB1"]

class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(THROTTLE_P, THROTTLE_I)
controller.set_desired(set_speed)

def prepare_model(model_path):
    from keras import __version__ as keras_version
    # check that model Keras version is same as local Keras version
    f = h5py.File(model_path, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    return load_model(model_path)

def preprocess(raw_img):
    # This should be similar to the one in drive.py
    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV)[:, :, 2]
    img = cv2.Sobel(img, -1, 0, 1, ksize=3)
    img = img / 255.0
    img = img > 0.5
    img = np.array([img])
    img = np.rollaxis(np.concatenate((img, img, img)), 0, 3)
    return img[:, :, [0]]


def main():
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
        help="Path to model definition h5 file. Model weights should be contained in the same path.")
    parser.add_argument('dir', type=str,
        help="Path to images and data to test the car with. " + \
        "This directory contains the following:\n" + \
        "- A directory named \"recorded\" that contains images the car will see.\n" + \
        "- A csv file for ground truth.\n\n" + \
        "Test results will then be created in this directory."
    )
    parser.add_argument('-d', action='store_true',
        help="When flag `-d` is included, send command to the actuators. " +
        "This is useful to inspect how the car runs when given input data.\n" +
        "DON'T FORGET TO SET THE CAR TO \"AUTO\" MODE.")
    parser.add_argument('-t', action='store_true',
        help="By default, this script does not actuate throttle. To allow it to"
        "send throttle commands, include flag `-t`.")

    args = parser.parse_args()
    allow_drive = args.d
    allow_throttle = args.t

    if allow_drive:
        port = choose_port(ports)

    # Prepare model
    model = prepare_model(args.model)

    # Setup test dir and all path related variables.
    test_dir = args.dir
    path = glob.glob(os.path.join(test_dir, 'recorded', '*.jpg'))
    path_r = os.path.split(path[0])
    test_images_dir = os.path.join(test_dir, "recorded")
    path = os.path.join(test_dir, '*.csv')
    test_image_csv = glob.glob(path)[0]
    test_result_dir = os.path.join(test_dir, 'test_results')
    test_result_imgs_dir = os.path.join(test_result_dir, 'images')

    os.makedirs(test_result_imgs_dir, exist_ok=True)

    # For debugging latency.
    previous_time = time.time()

    errors = 0.0
    speed = 0.0
    stats = {
        'id': [],
        'throttle': [],
        'steer': [],
        'error': [],
    }

    with open(test_image_csv, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        row_count = sum(1 for row in reader)
        csvfile.seek(0)
        next(reader, None)
        counter = 0
        print("row count:", row_count)
        for i, row in enumerate(reader):
            if allow_throttle:
                speed = controller.update(speed)
            else:
                speed = 0

            frame_path = os.path.join(test_images_dir, row[0])
            frame = cv2.imread(frame_path)
            final_img = preprocess(frame)
            img_array = np.asarray(final_img)[None, :, :, :]

            prediction = model.predict(\
                img_array, batch_size=1)

            new_steer = prediction[0][0]
            error = math.sqrt((float(row[1]) - new_steer)**2)
            errors += error
            stats['steer'].append(new_steer)
            stats['error'].append(error)
            stats['id'].append(row[0])
            stats['throttle'].append(speed)

            if allow_drive:
                if allow_throttle:
                    port.write(bytearray("{}{};".format(\
                        HOST_AUTO_THROTTLE.decode(), str(speed)), 'utf-8'))

                port.write(bytearray("{}{};".format(\
                    HOST_AUTO_STEER.decode(), str(new_steer)), 'utf-8'))


            # === Logging ===

            # Need to store and load again to work with one-channel image.
            f3 = np.array([final_img, final_img, final_img])
            f3 = np.rollaxis(f3, 0, 3)
            f3 = f3 * 255.0
            f3 = f3[TARGET_CROP[0][0]:(TARGET_HEIGHT-TARGET_CROP[0][1]),
                    TARGET_CROP[1][0]:(TARGET_WIDTH-TARGET_CROP[1][1]), 2]
            save_path = os.path.join(test_result_imgs_dir, row[0])
            cv2.imwrite(save_path, f3)
            f3 = cv2.imread(save_path)
            text1 = "pred: {}".format(new_steer)
            text2 = "truth: {}".format(row[1])
            f3 = cv2.putText(f3, text1, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 210))
            f3 = cv2.putText(f3, text2, (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 150))

            viz = np.concatenate((f3, frame), axis=0)
            save_path = os.path.join(test_result_imgs_dir, row[0])
            cv2.imwrite(save_path, viz)
            if allow_drive:
                cv2.imshow("RC", viz)
                cv2.waitKey(1)

            if i%(max(1,int(row_count/50))) == 0:
                counter+=1
                sys.stdout.write("\r{0}".format("="*counter))

    print("\n")
    print("Average RMSE: {}".format(errors/len(stats['error'])))
    print("Results saved at", test_result_dir)
    plt.plot(stats['error'])
    plt.title('model root mean squared error loss (avg. {%.2f})'.format(errors/len(stats['error'])))
    plt.ylabel('errors')
    plt.xlabel('time')
    plt.savefig(os.path.join(test_result_dir, 'errors.png'), bbox_inches='tight')
    plt.show()

    plt.plot(stats['steer'])
    plt.title('steer values over time')
    plt.ylabel('steer values')
    plt.xlabel('time')
    plt.savefig(os.path.join(test_result_dir, 'steer.png'), bbox_inches='tight')
    plt.show()

    plt.plot(stats['throttle'])
    plt.title('throttle values over time')
    plt.ylabel('throttle values')
    plt.xlabel('time')
    plt.savefig(os.path.join(test_result_dir, 'throttle.png'), bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
