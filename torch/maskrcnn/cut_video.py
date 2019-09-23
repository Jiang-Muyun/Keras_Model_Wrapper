import cv2
import numpy as np

size = (1920,1080)
cap = cv2.VideoCapture('maskrcnn/S2_Cars_night.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
font = cv2.FONT_HERSHEY_SIMPLEX
out = cv2.VideoWriter('maskrcnn/S2_Cars_night_cut.avi',fourcc, 30.0, size)

record = False
index = 0
while(cap.isOpened()):
    ret, frame_bgr = cap.read()
    frame_bgr = cv2.resize(frame_bgr,size)
    if not ret:
        break
    
    disp = cv2.resize(frame_bgr, (0,0) ,fx=0.5, fy=0.5)
    cv2.imshow('frame_bgr',disp)

    if record:
        out.write(frame_bgr)
        index += 1
        print(index)
        key = cv2.waitKey(20)
    else:
        key = cv2.waitKey(10)
    
    if 32 == key:
        record = not record
        print(record)
    if key == 27:
        break

out.release()