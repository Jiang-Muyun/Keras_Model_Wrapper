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

sess = tf.compat.v1.Session()
sess.as_default()
tf.compat.v1.keras.backend.set_session(sess)

from IPython.display import clear_output
from mask_rcnn.rcnn_warpper import *
clear_output()

frame_bgr = cv2.imread(voc_samples[0])

with Tock('interference'):
    detections = mask_rcnn_predict(frame_bgr)
    rcnn_overlap = mask_rcnn_plot(detections,frame_bgr)

plt.imshow(cv2.cvtColor(rcnn_overlap,cv2.COLOR_BGR2RGB))
plt.show()