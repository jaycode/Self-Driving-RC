# Adds steering angle to the images.
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import glob
import csv
import os
import cv2
import numpy as np
import pickle
import argparse
import copy
import sys

DATA_DIR = "C:\\Users\\teguh\\Dropbox\\Projects\\Robotics\\Self-Driving-RC-Data\\recorded-2017-06-01.1"
CALIBRATION_FILE = "C:\\Users\\teguh\\Dropbox\\Projects\\Robotics\\Self-Driving-RC\\Computer\\calibrations\\cal-elp.p"

dir_path = os.path.dirname(os.path.realpath(__file__))

# Path to Computer root directory
ROOT_DIR = os.path.realpath(os.path.join(dir_path, '..'))

sys.path.append(ROOT_DIR)
from libraries.helpers import configuration, preprocess

config = configuration()


STEER_FIELD_ID = 1

# Make sure the target shape is the same with the one in driver/main.py
# i.e. look for cams setup with variable CAP_PROP_FRAME_WIDTH and CAP_PROP_FRAME_HEIGHT.
TARGET_WIDTH = config['target_width']
TARGET_HEIGHT = config['target_height']
TARGET_CROP = config['target_crop']
STEER_MIN = config['steer_min']
STEER_MAX = config['steer_max']

def main():

    parser = argparse.ArgumentParser(description='Preprocess images')
    parser.add_argument('--data-dir', type=str,
        default=DATA_DIR,
        help="Path to directory that contains csv file and a directory of images.")
    parser.add_argument('--calibration-file', type=str,
        default=CALIBRATION_FILE,
        help="Path to calibration file e.g. `/home/user/cal-elp.p`.")
    args = parser.parse_args()
    data_dir = args.data_dir
    calibration_file = args.calibration_file

    lines = []
    with open( calibration_file, "rb" ) as pfile:
        cal = pickle.load(pfile)
    mtx = cal['mtx']
    dist = cal['dist']

    DATA_FILE = os.path.join(data_dir, "recorded.csv")
    with open(DATA_FILE) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for line in reader:
            lines.append(line)

    img_paths = glob.glob(os.path.join(data_dir, "recorded","*.jpg"))
    os.makedirs(os.path.join(data_dir, "w_steer"), exist_ok=True)

    steer_range = STEER_MAX - STEER_MIN
    steer_mid = (steer_range/2) + STEER_MIN

    for i, row in enumerate(lines):
        img_path = os.path.join(data_dir, "recorded", row[0])
        img = cv2.imread(img_path)
        # Preprocessing - Make sure to copy them to drive.py and test.py
        # raw_img = cv2.undistort(img, mtx, dist, None, mtx)
        raw_img = copy.copy(img)

        final_img = preprocess(raw_img)
        print(final_img.shape)

        steer_from_mid = float(row[STEER_FIELD_ID])-steer_mid
        measurement = steer_mid + steer_from_mid
        measurement_str = '%.1f' % measurement

        f3 = np.array([final_img, final_img, final_img])
        f3 = np.rollaxis(f3, 0, 3)
        f3 = f3 * 255.0
        f3 = f3[TARGET_CROP[0][0]:(TARGET_HEIGHT-TARGET_CROP[0][1]),
                TARGET_CROP[1][0]:(TARGET_WIDTH-TARGET_CROP[1][1]), 2]
        save_path = os.path.join(data_dir, "w_steer", row[0])
        cv2.imwrite(save_path, f3)
        f3 = cv2.imread(save_path)
        f3 = cv2.putText(f3, row[STEER_FIELD_ID], (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        raw_img = raw_img[TARGET_CROP[0][0]:(TARGET_HEIGHT-TARGET_CROP[0][1]),
                TARGET_CROP[1][0]:(TARGET_WIDTH-TARGET_CROP[1][1]), :]
        viz = np.concatenate((f3, raw_img), axis=0)
        cv2.imwrite(save_path, viz)

        print("saved {}".format(save_path))


if __name__ == "__main__":
    main()