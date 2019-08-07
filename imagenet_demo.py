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

from libs.common import *
from libs.classification import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
sess.as_default()
tf.compat.v1.keras.backend.set_session(sess)

from IPython.display import clear_output
clear_output()

# m = Model_Warper(sess,'resnet50')
# m = Model_Warper(sess,'mobilenet_v2')
# m = Model_Warper(sess,'xception')
m = Model_Warper(sess,'inception_v3')
# m = Model_Warper(sess,'vgg16')
# m = Model_Warper(sess,'vgg19')
clear_output()

for fn in imagenet_samples:
    with Tick():
        prediction = m.predict(fn)
    print(top_n(prediction,n=5),'\n')