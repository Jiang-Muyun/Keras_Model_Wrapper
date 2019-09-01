# %matplotlib inline
import os
import time
import cv2
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
import keras
import tensorflow as tf
from tensorflow.python.keras import backend as K

from model_wrapper.utils import sub_plot,Tick,voc
from mask_rcnn.warpper import predict,plot

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='images', help='images or video demo')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    return parser.parse_args()

def images_demo(args):
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

def video_demo(args):
    cap = cv2.VideoCapture('data/demo_video.mp4')
    while(cap.isOpened()):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        with Tick('interference'):
            detections = predict(frame_bgr)
            overlap = plot(detections,frame_bgr)

        cv2.imshow('overlap',overlap)
        if 27 == cv2.waitKey(1):
            break

if __name__ == '__main__':
    args = parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if args.mode == 'images':
        images_demo(args)
    if args.mode == 'video':
        video_demo(args)