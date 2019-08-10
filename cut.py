import cv2
cap = cv2.VideoCapture('tmp/9_very_close_takeoffs_landings_qxnBcZ0CETg.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('tmp/cut.avi',fourcc, 20.0, (960,540))

counter = 0
while(cap.isOpened()):
    ret, frame_bgr = cap.read()
    counter += 1
    if counter < 200:
        continue
    frame_bgr = cv2.resize(frame_bgr,(0,0),fx=0.5,fy=0.5)
    if counter % 5 == 0:
        out.write(frame_bgr)
    #cv2.imshow('frame_bgr',frame_bgr)
    print(counter,frame_bgr.shape)
    #if 27 == cv2.waitKey(1):
    #    break
    if counter == 650:
        break
out.release()