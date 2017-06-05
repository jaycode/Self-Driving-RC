import numpy as np
import cv2

cap = cv2.VideoCapture(0)

index = 0
while (index<1000):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    #cv2.imshow('frame',frame)
    index +=1
    cv2.imwrite('data/frame'+str(index)+".jpg",frame)

