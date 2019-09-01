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
import keras
import tensorflow as tf
from tensorflow.python.keras import backend as K

from deeplab.model import Deeplabv3
from model_wrapper.utils import auto_download

class Deeplab_Wrapper():
    def __init__(self, sess, model_name, dataset = 'pascal_voc', save_and_reload = False):
        self.sess = sess
        self.model_name = model_name
        self.dataset = dataset
        self.save_and_reload = save_and_reload
        self.folder = 'tmp/weights/deeplab/'

        assert model_name in ['xception','mobilenetv2'],'Unsupported name:'+ model_name
        assert dataset in ['pascal_voc','cityscapes'],'Unsupported dataset:'+ dataset
        
        self.load_model()
        self.predict(np.zeros((512,512,3),dtype=np.float32))
        print('> Done')
        
    def load_model(self):
        tag = 'deeplab_' + self.model_name + '_' + self.dataset
    
        print('> Loading Model [%s] ' % (self.model_name))
        with self.sess.as_default():
            with tf.variable_scope('model'):
                self.model = Deeplabv3(
                    weights = self.dataset, 
                    backbone = self.model_name, 
                    weights_path = auto_download(self.folder,tag)
                )
                
                if self.save_and_reload:
                    # Save and reload the model to avoid batch normalization issues
                    model_weights = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='model')
                    print('> Saving Model')
                    tf.compat.v1.train.Saver(model_weights).save(self.sess, 'tmp/')
                    print('> Reload Model')
                    tf.compat.v1.train.Saver(model_weights).restore(self.sess, 'tmp/')
        
        self.model_in = self.model.layers[0].input
        self.model_out = self.model.layers[-1].output
        self.model_softmax = tf.compat.v1.nn.softmax(self.model_out,-1)
        self.model_label_out = tf.cast(tf.compat.v1.math.argmax(self.model_out,-1),tf.uint8)
        
    def resize_keeping_aspect_ratio(self, img, dsize=(512, 512), inter=cv2.INTER_AREA):
        self.src_shape = img.shape
        rows, cols, channals = img.shape
        max_dim = max(rows, cols)
        tmp = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        tmp[:rows, :cols, :] = img
        resized = cv2.resize(tmp, dsize, interpolation=inter)
        return resized

    # def resize_back(self, img, dsize = None, BGR = False, inter=cv2.INTER_AREA):
    #     if dsize == None:
    #         rows, cols, channals = self.src_shape
    #     max_dim = max(rows, cols)
    #     resized = cv2.resize(img, (max_dim,max_dim), interpolation=inter)
    #     img = resized[:rows, :cols, :]
    #     if BGR:
    #         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     return img

    def resize_back(self, img, dsize = None, inter=cv2.INTER_AREA):
        if dsize == None:
            rows, cols, channals = self.src_shape
        max_dim = max(rows, cols)
        resized = cv2.resize(img, (max_dim,max_dim), interpolation=inter)
        img = resized[:rows, :cols, :]
        return img

    def load_image(self, fn, dsize=(512, 512)):
        if isinstance(fn, str):
            fn = [fn]
            input_type = 'single'
        elif isinstance(fn,list):
            input_type = 'batch'
        else:
            raise ValueError('Unknown input type' + fn)
        
        buf = []
        for path in fn:
            assert os.path.exists(path),path
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.resize_keeping_aspect_ratio(image, dsize)
            buf.append(image)
        image_batch = np.array(buf, dtype=np.uint8)
        if input_type == 'single':
            return image_batch[0]
        else:
            return image_batch

    def project(self,x):
        return (x.astype(np.float32) / 127.5) - 1.0

    def reproject(self,x):
        assert x.dtype == np.float32 or x.dtype == np.float64
        return ((x + 1.0) * 127.5).astype(np.uint8)
    
    def predict(self, input_var, output_mode = 'label'):
        if isinstance(input_var, str):
            img_batch = self.project(self.load_image([input_var]))
        elif isinstance(input_var, list):
            img_batch = self.project(self.load_image(input_var))
        elif isinstance(input_var, np.ndarray) and input_var.shape == (512, 512, 3):
            img_batch = np.expand_dims(input_var, axis=0)
        elif isinstance(input_var, np.ndarray) and input_var.shape[1:] == (512, 512, 3):
            img_batch = input_var
        else:
            raise ValueError('Unknown input '+str(input_var))

        if output_mode == 'label':
            return self.sess.run(self.model_label_out, {self.model_in: img_batch})
        if output_mode == 'softmax':
            return self.sess.run(self.model_softmax, {self.model_in: img_batch})
