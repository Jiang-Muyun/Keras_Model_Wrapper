# %matplotlib inline
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import cv2
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import random
import keras
import tensorflow as tf
from tensorflow.python.keras import backend as K

from utils import voc,sub_plot,Tick,new_session
from deeplab.warpper import Deeplab_Wrapper

sess = new_session()
xception = Deeplab_Wrapper(sess,'xception')
mobilenet = Deeplab_Wrapper(sess,'mobilenetv2')
from mask_rcnn.warpper import predict,plot

cap = cv2.VideoCapture('tmp/videos/9_Very_Close_Takeoffs_Landings.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
font = cv2.FONT_HERSHEY_SIMPLEX
out = cv2.VideoWriter('tmp/deeplab_cmp.avi',fourcc, 30.0, (640*2,360*2))
counter,filter1,filter2,filter3 = 0,0,0,0

while(cap.isOpened()):
    with Tick(str(counter)):
        ret, frame_bgr = cap.read()
        frame_bgr = cv2.resize(frame_bgr,(0,0),fx=0.5,fy=0.5)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_reshape = mobilenet.resize_keeping_aspect_ratio(frame_rgb)
            
        with Tock('xception') as t1:
            label_x = xception.predict(xception.project(img_reshape))
            x_disp = mobilenet.resize_back(voc.get_label_colormap(label_x[0]))
            x_overlap = cv2.addWeighted(frame_bgr, 0.5, x_disp, 0.5, 20)
        
        with Tock('mobilenetv2') as t2:
            label_m = mobilenet.predict(mobilenet.project(img_reshape))
            m_disp = mobilenet.resize_back(voc.get_label_colormap(label_m[0]))
            m_overlap = cv2.addWeighted(frame_bgr, 0.5, m_disp, 0.5, 20)
        
        with Tock('mask_rcnn') as t3:
            detections = predict(frame_bgr)
            rcnn_overlap = plot(detections,frame_bgr)


        filter1 = filter1 * 0.8 + t1.fps * 0.2
        filter2 = filter2 * 0.8 + t2.fps * 0.2
        filter3 = filter3 * 0.8 + t3.fps * 0.2
        
        x_text = 'Deeplab Xception: %2.0f fps'%(filter1)
        m_text = 'Deeplab MobileNetv2: %2.0f fps'%(filter2)
        rcnn_text = 'MaskRCNN ResNet101: %2.0f fps'%(filter3)
        
        cv2.putText(frame_bgr,'Camera',(10,40), font, 1,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(x_overlap,x_text,(10,40), font, 1,(0,255,0),2,cv2.LINE_AA)
        cv2.putText(m_overlap,m_text,(10,40), font, 1,(0,255,0),2,cv2.LINE_AA)
        cv2.putText(rcnn_overlap,'Mask-RCNN',(10,40), font, 1,(0,255,0),2,cv2.LINE_AA)
        out.write(np.vstack((np.hstack((frame_bgr,x_overlap)),np.hstack((m_overlap,rcnn_overlap)))))
        
        counter += 1
        if counter == 10:
            break

cap.release()
out.release()