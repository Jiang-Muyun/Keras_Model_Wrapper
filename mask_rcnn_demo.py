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

for fn in glob.glob('data/COCO/*'):
    frame_bgr = cv2.imread(fn)
    with Tick('interference'):
        detections = predict(frame_bgr)
        rcnn_overlap = plot(detections,frame_bgr)

    fig = plt.figure(figsize=(12, 4), dpi=100, facecolor='w', edgecolor='k')
    sub_plot(fig,1,2,1,'image',cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB))
    sub_plot(fig,1,2,2,'overlap',cv2.cvtColor(rcnn_overlap,cv2.COLOR_BGR2RGB))
    plt.show(block = False)

plt.show()