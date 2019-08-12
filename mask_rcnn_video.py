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

from model_wrapper.utils import sub_plot,Tick,voc
from mask_rcnn.warpper import predict,plot

cap = cv2.VideoCapture('data/demo_video.mp4')

while(cap.isOpened()):
    ret, frame_bgr = cap.read()
    if not ret:
        break
    
    with Tick('interference'):
        detections = predict(frame_bgr)
        overlap = plot(detections,frame_bgr)

    cv2.imshow('overlap',overlap)
    if 27 == cv2.waitKey(1):
        break