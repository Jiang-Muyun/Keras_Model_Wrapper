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
from IPython.display import clear_output
import tensorflow as tf
from tensorflow.python.keras import backend as K

from model_wrapper.utils import *
from model_wrapper.segmentation import *

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
sess.as_default()
tf.compat.v1.keras.backend.set_session(sess)

from mask_rcnn.rcnn_warpper import *
clear_output()

cap = cv2.VideoCapture(http_download('tmp/videos',domain + files['videos'][0]))

while(cap.isOpened()):
    ret, frame_bgr = cap.read()
    frame_bgr = cv2.resize(frame_bgr,(0,0),fx=0.5,fy=0.5)

    with Tick('interference'):
        detections = mask_rcnn_predict(frame_bgr)
        overlap = mask_rcnn_plot(detections,frame_bgr)

    cv2.imshow('overlap',overlap)
    if 27 == cv2.waitKey(1):
        break