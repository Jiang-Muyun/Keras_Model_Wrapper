import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import time
import glob
import numpy as np
import matplotlib.pyplot as plt

from model_wrapper.utils import Tick,new_session
from model_wrapper.classification import Model_Wrapper,top_n

wrapper = Model_Wrapper(new_session(),'mobilenet_v2_1.0')
wrapper.print_support_models()

for fn in glob.glob('data/ImageNet/*'):
    with Tick('interference'):
        prediction = wrapper.predict(fn)
        print('\n'+top_n(prediction,n=5))