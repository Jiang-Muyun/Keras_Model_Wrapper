# %matplotlib inline
import os
import sys
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

from model_wrapper.utils import voc,sub_plot,Tick,new_session
from deeplab.warpper import Deeplab_Wrapper

assert sys.argv[1] in ['mobilenetv2','xception'], sys.argv[1]
wrapper = Deeplab_Wrapper(new_session(),sys.argv[1])

voc.show_legend()
for fn in glob.glob('data/Pascal_Voc/*'):
    img = wrapper.resize_back(wrapper.load_image(fn))
    with Tick('interference'):
        label = wrapper.predict(fn)[0]
    disp = wrapper.resize_back(voc.get_label_colormap(label))
    overlap = cv2.addWeighted(img, 0.5, disp, 0.5, 20)

    fig = plt.figure(figsize=(12, 4), dpi=100, facecolor='w', edgecolor='k')
    sub_plot(fig,1,3,1,'image',img)
    sub_plot(fig,1,3,2,voc.semantic_report(label),disp)
    sub_plot(fig,1,3,3,'overlap',overlap)
    plt.show(block = False)

plt.show()