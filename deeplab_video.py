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

from libs.common import *
from libs.segmentation import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
sess.as_default()
tf.compat.v1.keras.backend.set_session(sess)

from IPython.display import clear_output
clear_output()

# m = Segmentation_Wrapper(sess,'mobilenetv2')
m = Segmentation_Wrapper(sess,'xception')
clear_output()

fn_video = 'tmp/videos/9_Very_Close_Takeoffs_Landings.mp4'
assert os.path.exists(fn_video)
cap = cv2.VideoCapture(fn_video)

while(cap.isOpened()):
    ret, frame_bgr = cap.read()
    frame_bgr = cv2.resize(frame_bgr,(0,0),fx=0.5,fy=0.5)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_reshape = m.resize_keeping_aspect_ratio(frame_rgb)
    
    with Tick():
        label = m.predict(m.project(img_reshape))
        disp = m.resize_back(voc.get_label_colormap(label[0]))
        overlap = cv2.addWeighted(frame_bgr, 0.5, disp, 0.5, 20)
    
    cv2.imshow('overlap',overlap)
    if 27 == cv2.waitKey(1):
        break