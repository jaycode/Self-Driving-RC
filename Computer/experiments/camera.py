import numpy as np
import cv2

cam = cv2.VideoCapture(1)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 2000)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

index = 0
while (index<1000):
    # Capture frame-by-frame
    ret, frame = cam.read()

    cv2.imshow("RC", frame)
    print(frame.shape)
    cv2.waitKey(1)
