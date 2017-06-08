import numpy as np
import cv2
import glob
import os

ORIGINAL_DIR = 'C:\\Users\\teguh\\Dropbox\\Projects\\Robotics\\Self-Driving-RC\\Computer\\experiments\\calibration-elp'
RESIZED_DIR = 'C:\\Users\\teguh\\Dropbox\\Projects\\Robotics\\Self-Driving-RC\\Computer\\experiments\\calibration-elp-final'

# Original calibration images
images = glob.glob(os.path.join(ORIGINAL_DIR, "*.jpg"))

os.makedirs(RESIZED_DIR, exist_ok=True)

print(images)
for image_path in images:
    img = cv2.imread(image_path)
    res = cv2.resize(img,(320, 240))
    name = os.path.basename(image_path)
    path = os.path.join(RESIZED_DIR, name)
    cv2.imwrite(path, res)
    print("Created", path)