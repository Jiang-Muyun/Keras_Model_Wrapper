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
clear_output()

# m = Segmentation_Wrapper(sess,'mobilenetv2')
m = Segmentation_Wrapper(sess,'xception')
clear_output()

voc.show_legend()
for fn in voc_samples:
    with Tick():
        label = m.predict(fn)[0]
    disp = m.resize_back(voc.get_label_colormap(label))
    img = m.resize_back(m.load_image(fn))

    fig = plt.figure(dpi=80)
    sub_plot(fig,1,2,1,'image',img)
    sub_plot(fig,1,2,2,voc.semantic_report(label),disp)
    plt.show(block = False)

plt.show()