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

DATA_DIR = "C:\\Users\\teguh\\Dropbox\\Projects\\Robotics\\Self-Driving-RC-Data\\recorded-2017-06-01.1"
CALIBRATION_FILE = "C:\\Users\\teguh\\Dropbox\\Projects\\Robotics\\Self-Driving-RC\\Computer\\calibrations\\cal-elp.p"

STEER_MIN = 30
STEER_MAX = 993
STEER_FIELD_ID = 1
TARGET_WIDTH = 320
TARGET_HEIGHT = 240
TARGET_CROP = ((60, 20), (0, 0))

def main():
    lines = []

    with open( CALIBRATION_FILE, "rb" ) as pfile:
        cal = pickle.load(pfile)
    mtx = cal['mtx']
    dist = cal['dist']

    DATA_FILE = "{}\\recorded.csv".format(DATA_DIR)
    with open(DATA_FILE) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for line in reader:
            lines.append(line)

    img_paths = glob.glob("{}\\recorded\\*.jpg".format(DATA_DIR))
    os.makedirs("{}\\w_steer".format(DATA_DIR), exist_ok=True)

    steer_range = STEER_MAX - STEER_MIN
    steer_mid = (steer_range/2) + STEER_MIN

    for i, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        # Preprocessing - Make sure to copy them to drive.py and test.py
        undist_img = cv2.undistort(img, mtx, dist, None, mtx)

        img = cv2.cvtColor(undist_img, cv2.COLOR_BGR2HSV)[:, :, 2]
        img = cv2.Sobel(img, -1, 0, 1, ksize=3)
        img = img / 255.0
        img = img > 0.5

        img1 = cv2.imread(img_path)
        undist_img = cv2.undistort(img1, mtx, dist, None, mtx)
        img1 = cv2.cvtColor(undist_img, cv2.COLOR_BGR2HSV)
        img1 = cv2.inRange(img1, (62, 28, 135), (145, 255, 190))
        img1 = img1 / 255
        img1 = img1 > 0.5
        final_img = (img==1) | (img1==1)

        steer_from_mid = float(lines[i][STEER_FIELD_ID])-steer_mid
        measurement = steer_mid + steer_from_mid
        measurement_str = '%.1f' % measurement
        print("repeated:")
        f3 = np.array([final_img, final_img, final_img])
        f3 = np.rollaxis(f3, 0, 3)
        f3 = f3 * 255
        print(f3.shape)
        f3 = cv2.putText(f3, lines[i][STEER_FIELD_ID], (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        f3 = f3[TARGET_CROP[0][0]:(TARGET_HEIGHT-TARGET_CROP[0][1]),
                TARGET_CROP[1][0]:(TARGET_WIDTH-TARGET_CROP[1][1]), 2]
        cv2.imwrite("{}\\w_steer\\{}".format(DATA_DIR, lines[i][0]), f3)
        print("saved {}".format("{}\\w_steer\\{}".format(DATA_DIR, lines[i][0])))

if __name__ == "__main__":
    main()