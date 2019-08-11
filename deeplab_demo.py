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

# wrapper = Segmentation_Wrapper(sess,'mobilenetv2')
wrapper = Segmentation_Wrapper(sess,'xception')
clear_output()

voc.show_legend()
for fn in glob.glob('data/Pascal_Voc/*'):
    with Tick('interference'):
        label = wrapper.predict(fn)[0]
    disp = wrapper.resize_back(voc.get_label_colormap(label))
    img = wrapper.resize_back(wrapper.load_image(fn))
    overlap = cv2.addWeighted(img, 0.5, disp, 0.5, 20)

    fig = plt.figure(figsize=(12, 4), dpi=100, facecolor='w', edgecolor='k')
    sub_plot(fig,1,3,1,'image',img)
    sub_plot(fig,1,3,2,voc.semantic_report(label),disp)
    sub_plot(fig,1,3,3,'overlap',overlap)
    plt.show(block = False)

plt.show()