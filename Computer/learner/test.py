# To use, run:
# `sudo su`
# `python test.py [--models model-paths] [--dir test-dir-path] [-d] [-t]`

# === Model Paths ===
# `models` is the list of paths to model definition h5.
# Model definition is created by learner/learn.py script.
# The resulting prediction is going to be the average of all of the
# predictions.

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
from libraries.helpers import configuration, choose_port, preprocess, prepare_model

config = configuration()

# This is the smallest current camera may support.
# (i.e. setting CAP_PROP_FRAME_WIDTH and HEIGHT smaller than this won't help)
TARGET_WIDTH = config['target_width']
TARGET_HEIGHT = config['target_height']
TARGET_CROP = config['target_crop']

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

def main():
    models = []
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('dir', type=str,
        help="Path to images and data to test the car with. " + \
        "This directory contains the following:\n" + \
        "- A directory named \"recorded\" that contains images the car will see.\n" + \
        "- A csv file for ground truth.\n\n" + \
        "Test results will then be created in this directory."
    )
    parser.add_argument('--models', type=str,
        nargs='+',
        help="list of paths to model definition h5. " + \
        "Model definition is created by learner/learn.py script. " + \
        "The resulting prediction is going to be the average of all of the " + \
        "predictions.")
    parser.add_argument('-d', action='store_true',
        default=False,
        help="When flag `-d` is included, send command to the actuators. " +
        "This is useful to inspect how the car runs when given input data.\n" +
        "DON'T FORGET TO SET THE CAR TO \"AUTO\" MODE.")
    parser.add_argument('-t', action='store_true',
        default=False,
        help="By default, this script does not actuate throttle. To allow it to"
        "send throttle commands, include flag `-t`.")

    args = parser.parse_args()
    allow_drive = args.d
    allow_throttle = args.t

    if allow_drive:
        port = choose_port(ports)

    # Prepare models
    if not args.models:
        raise Exception("[--models] parameter is needed")
    else:
        for model_path in args.models:
            models.append(prepare_model(model_path))

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

    # total squared error
    t_se = 0.0
    # total root squared error
    t_rse = 0.0
    speed = 0.0
    stats = {
        'id': [],
        'throttle': [],
        'steer': [],
        'rse': [],
        'se': []
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
            speed = controller.update(speed)

            frame_path = os.path.join(test_images_dir, row[0])
            frame = cv2.imread(frame_path)
            final_img = preprocess(frame, TARGET_HEIGHT, TARGET_WIDTH, TARGET_CROP)
            img_array = np.asarray(final_img)[None, :, :, :]

            new_steer_total = 0
            for model in models:
                prediction = model.predict(\
                    img_array, batch_size=1)
                new_steer_total += prediction[0][0]
            new_steer = new_steer_total / len(models)
            # squared error
            se = (float(row[1]) - new_steer)**2
            # root squared error
            rse = math.sqrt(se)
            t_se += se
            t_rse += rse
            stats['steer'].append(new_steer)
            stats['se'].append(se)
            stats['rse'].append(rse)
            stats['id'].append(row[0])
            stats['throttle'].append(speed)

            if allow_drive:
                if allow_throttle:
                    port.write(bytearray("{}{};".format(\
                        HOST_AUTO_THROTTLE.decode(), str(speed)), 'utf-8'))

                port.write(bytearray("{}{};".format(\
                    HOST_AUTO_STEER.decode(), str(new_steer)), 'utf-8'))


            # === Logging ===

            f3 = np.stack((final_img[:, :, 0], final_img[:, :, 0], final_img[:, :, 0]), axis=2)
            f3 = (f3 * 255.0).astype(np.uint8)

            text1 = "pred: {}".format(new_steer)
            text2 = "truth: {}".format(row[1])
            text3 = "Squared-Error: {0:.2f}".format(se)
            text4 = "Root-Squared-Error: {0:.2f}".format(rse)
            f3 = cv2.putText(f3, text1, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 210))
            f3 = cv2.putText(f3, text2, (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 150))
            f3 = cv2.putText(f3, text3, (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 50, 255))
            f3 = cv2.putText(f3, text4, (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 50, 255))

            viz = np.concatenate((f3, frame), axis=0)

            save_path = os.path.join(test_result_imgs_dir, row[0])
            cv2.imwrite(save_path, viz)
            if allow_drive:
                cv2.imshow("RC", viz)
                cv2.waitKey(1)

            if i%(max(1,int(row_count/40))) == 0:
                counter+=1
                sys.stdout.write("\r{0}".format("="*counter))

    print("\n")
    print("MSE: {0:.2f}".format(t_se/len(stats['se'])))
    print("RMSE: {0:.2f}".format(t_rse/len(stats['rse'])))
    print("Results saved at", test_result_dir)
    plt.plot(stats['se'])
    plt.title('model squared error loss (mean: {0:.2f})'.format(t_se/len(stats['se'])))
    plt.ylabel('squared error')
    plt.xlabel('time')
    plt.savefig(os.path.join(test_result_dir, 'MSE.png'), bbox_inches='tight')
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
