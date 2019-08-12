# %matplotlib inline
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
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

from model_wrapper.utils import voc,sub_plot,Tick,new_session
from deeplab.warpper import Deeplab_Wrapper

assert sys.argv[1] in ['mobilenetv2','xception'], sys.argv[1]
wrapper = Deeplab_Wrapper(new_session(),sys.argv[1])

cap = cv2.VideoCapture('data/demo_video.mp4')

while(cap.isOpened()):
    ret, frame_bgr = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_reshape = wrapper.resize_keeping_aspect_ratio(frame_rgb)
    
    with Tick('interference'):
        label = wrapper.predict(wrapper.project(img_reshape))
        disp = wrapper.resize_back(voc.get_label_colormap(label[0]))
        overlap = cv2.addWeighted(frame_bgr, 0.5, disp, 0.5, 20)
    
    cv2.imshow('overlap',overlap)
    if 27 == cv2.waitKey(1):
        break