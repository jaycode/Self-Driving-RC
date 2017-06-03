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

    for i, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
        img = cv2.putText(img,lines[i][1], (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        img = img[:, :, :]

        cv2.imwrite("{}\\w_steer\\{}".format(DATA_DIR, lines[i][0]), img)
        print("saved {}".format("{}\\w_steer\\{}".format(DATA_DIR, lines[i][0])))

if __name__ == "__main__":
    main()