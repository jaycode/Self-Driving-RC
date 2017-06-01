# Find the right cropping and color channel.
import cv2
import numpy as np

IMG = "C:\\Users\\teguh\\Dropbox\\Projects\\Robotics\\Self-Driving-RC-Data\\recorded-2017-05-31\\recorded\\2017-06-01_01-53-40.616923.jpg"

def main():
    img = cv2.imread(IMG)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    img = img[70:, :, 1]

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()