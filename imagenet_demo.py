# %matplotlib inline
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras.models import Model
from keras.applications import resnet50, mobilenet_v2, inception_v3, xception
from tensorflow.python.keras import backend as K
from IPython.display import clear_output

from model_wrapper.utils import *
from model_wrapper.classification import *

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
sess.as_default()
tf.compat.v1.keras.backend.set_session(sess)

"""
Please select one of the following models
    'resnet50', 
    'mobilenet_v2_0.35', 
    'mobilenet_v2_0.5', 
    'mobilenet_v2_0.75', 
    'mobilenet_v2_1.0', 
    'mobilenet_v2_1.3', 
    'mobilenet_v2_1.4', 
    'xception', 
    'inception_v3', 
    'inception_resnet_v2', 
    'vgg16', 
    'vgg19', 
    'densenet121', 
    'densenet169', 
    'densenet201', 
    'nasnet-mobile', 
    'nasnet-large'
"""

wrapper = Model_Wrapper(sess,'mobilenet_v2_1.0')

for fn in imagenet_samples:
    with Tick('interference'):
        prediction = wrapper.predict(fn)
        print()
        print(top_n(prediction,n=5))