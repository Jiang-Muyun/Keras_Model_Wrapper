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

from lib.utils import *
from lib.segmentation import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
sess.as_default()
tf.compat.v1.keras.backend.set_session(sess)

from mask_rcnn.rcnn_warpper import *
clear_output()

for fn in voc_samples:
    frame_bgr = cv2.imread(fn)
    with Tick('interference'):
        detections = mask_rcnn_predict(frame_bgr)
        rcnn_overlap = mask_rcnn_plot(detections,frame_bgr)

    fig = plt.figure(figsize=(12, 4), dpi=100, facecolor='w', edgecolor='k')
    sub_plot(fig,1,2,1,'image',cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB))
    sub_plot(fig,1,2,2,'overlap',cv2.cvtColor(rcnn_overlap,cv2.COLOR_BGR2RGB))
    plt.show(block = False)

plt.show()