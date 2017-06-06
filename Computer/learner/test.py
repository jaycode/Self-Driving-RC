# To use, run:
# `sudo su`
# `source activate python3`
# `python test.py [model-path] [test-dir-path]`

# Set `dir` to a directory that contains the following:
# - A directory with images to see how the car drives with these data.
# - A csv file for ground truth.

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

# This is the smallest current camera may support.
# (i.e. setting CAP_PROP_FRAME_WIDTH and HEIGHT smaller than this won't help)
TARGET_WIDTH = 320
TARGET_HEIGHT = 240

DEFAULT_MODEL = "C:\\Users\\teguh\\Dropbox\\Projects\\Robotics\\Self-Driving-RC-Data\\recorded-2017-06-01.1\\model.h5"
DEFAULT_DIR = "C:\\Users\\teguh\\Dropbox\\Projects\\Robotics\\Self-Driving-RC-Data\\recorded-2017-06-01.1"
# TODO: This is currently throttle value but we will update it once we got
#       accelerometer.
set_speed = 130

MIN_THROTTLE = 50
MAX_THROTTLE = 130
THROTTLE_P=0.3
THROTTLE_I=0.08

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

def main():
    print("Listening for commands...");

    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('--model', type=str,
        default=DEFAULT_MODEL,
        help="Path to model definition json. Model weights should be on the same path.")
    parser.add_argument('--dir', type=str,
        default=DEFAULT_DIR,
        help="Path to images and data to test the car with. " + \
    "This directory contains the following:\n" + \
    "- A directory with images to see how the car drives with these data.\n" + \
    "- A csv file for ground truth."
    )

    args = parser.parse_args()

    model = prepare_model(args.model)

    test_dir = args.dir
    path = glob.glob(os.path.join(test_dir, '*', '*.jpg'))
    path_r = os.path.split(path[0])
    test_images_dir = path_r[0:len(path_r)-1][0]
    path = os.path.join(test_dir, '*.csv')
    test_image_csv = glob.glob(path)[0]
    test_result_dir = os.path.join(test_dir, 'results')
    os.makedirs(test_result_dir, exist_ok=True)

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
            frame = cv2.imread(os.path.join(test_images_dir, row[0]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

            image_array = np.asarray(frame)
            speed = controller.update(speed)
            prediction = model.predict(\
                image_array[None, :, :, :], batch_size=1)
            new_steer = prediction[0][0]
            error = math.sqrt((float(row[1]) - new_steer)**2)
            errors += error
            stats['steer'].append(new_steer)
            stats['error'].append(error)
            stats['id'].append(row[0])
            stats['throttle'].append(speed)
            if i%(int(row_count/100)) == 0:
                counter+=1
                sys.stdout.write("\r{0}".format("="*counter))

    print("\n")
    print("Errors: {}".format(errors))
    print("Results saved at", test_result_dir)
    plt.plot(stats['error'])
    plt.title('model mean squared error loss')
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
