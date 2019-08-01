%matplotlib inline
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
from libs.pascal_voc import *
from libs.semantic_warper import *

sess = tf.compat.v1.Session()
sess.as_default()
tf.compat.v1.keras.backend.set_session(sess)

from IPython.display import clear_output
clear_output()

x = Segmentation_Wrapper(sess,'xception')
m = Segmentation_Wrapper(sess,'mobilenetv2')
base = '/media/james/Ubuntu_Data/dataset/airplane/'
cap = cv2.VideoCapture(os.path.join(base,'9_Very_Close_Takeoffs_Landings.mp4'))

while(cap.isOpened()):
    with Tick(''):
        with Tock('Pre-Processing'):
            ret, frame_bgr = cap.read()
            frame_bgr = cv2.resize(frame_bgr,(0,0),fx=0.5,fy=0.5)
            img_reshape = m.resize_keeping_aspect_ratio(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            
        with Tock('xception'):
            label_x = x.predict(x.project(img_reshape))
            x_disp = m.resize_back(pascal_voc.get_label_colormap(label_x[0]))
            
        with Tock('mobilenetv2'):
            label_m = m.predict(m.project(img_reshape))
            m_disp = m.resize_back(pascal_voc.get_label_colormap(label_m[0]))
            
        with Tock('Post-Processing'):
            cv2.imshow('img',frame_bgr)
            cv2.imshow('semantic_x',x_disp)
            cv2.imshow('semantic_m',m_disp)
            if 27 == cv2.waitKey(1):
                cv2.destroyAllWindows()
                break