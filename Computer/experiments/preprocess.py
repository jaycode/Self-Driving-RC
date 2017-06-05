# Adds steering angle to the images.
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import glob
import csv
import os
import cv2
import numpy as np


DATA_DIR = "C:\\Users\\teguh\\Dropbox\\Projects\\Robotics\\Self-Driving-RC-Data\\recorded-2017-06-01.1"

STEER_MIN = 30
STEER_MAX = 993
STEER_FIELD_ID = 1
TARGET_WIDTH = 320
TARGET_HEIGHT = 240
TARGET_CROP = ((70, 20), (0, 0))

def main():
    lines = []

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
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)

        steer_from_mid = float(lines[i][STEER_FIELD_ID])-steer_mid
        measurement = steer_mid + steer_from_mid
        measurement_str = '%.1f' % measurement
        
        img = cv2.putText(img, lines[i][STEER_FIELD_ID], (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        img = img[TARGET_CROP[0][0]:(TARGET_HEIGHT-TARGET_CROP[0][1]),
                  TARGET_CROP[1][0]:(TARGET_WIDTH-TARGET_CROP[1][1]), :]

        cv2.imwrite("{}\\w_steer\\{}".format(DATA_DIR, lines[i][0]), img)
        print("saved {}".format("{}\\w_steer\\{}".format(DATA_DIR, lines[i][0])))

if __name__ == "__main__":
    main()