# %matplotlib inline
import os
import sys
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

from model_wrapper.utils import voc,sub_plot,Tick,new_session
from deeplab.warpper import Deeplab_Wrapper

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='xception', help='mobilenetv2 or xception')
    parser.add_argument('--dataset', default='pascal_voc', help='pascal_voc or cityscapes, cityscapes is buggy')
    parser.add_argument('--mode', default='images', help='image or video demo')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    return parser.parse_args()

def images_demo(args):
    wrapper = Deeplab_Wrapper(new_session(),args.model_name,args.dataset)
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

def video_demo(args):
    cap = cv2.VideoCapture('data/demo_video.mp4')
    wrapper = Deeplab_Wrapper(new_session(),args.model_name,args.dataset)
    while(cap.isOpened()):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_reshape = wrapper.resize_keeping_aspect_ratio(frame_rgb)
        
        with Tick('interference'):
            label = wrapper.predict(wrapper.project(img_reshape))
            disp = wrapper.resize_back(voc.get_label_colormap(label[0]))
            overlap = cv2.addWeighted(frame_bgr, 0.5, disp, 0.5, 20)
        
        cv2.imshow('overlap',overlap)
        if 27 == cv2.waitKey(1):
            break

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if args.mode == 'images':
        images_demo(args)
    if args.mode == 'video':
        video_demo(args)
