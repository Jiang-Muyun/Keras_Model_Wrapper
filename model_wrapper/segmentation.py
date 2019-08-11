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

from model_wrapper.utils import *
from deeplab.model import Deeplabv3

class Pascal_Voc_Utill():
    def __init__(self):
        moduleBase = os.path.abspath(os.path.join(os.path.realpath(__file__), '../../'))
        tmp = json.load(open(os.path.join(moduleBase,'data/pascal_voc.json'), 'r'))
        self.num_classes = tmp['num_classes']
        self.labels = tmp['labels_short']
        self.labels_index = tmp['labels_index']
        self.colors = tmp['colors']
        self.colormap = np.array(self.colors, dtype=np.uint8)

    def get_label_colormap(self,label):
        assert label.dtype in [np.uint8, np.uint16, np.uint32], label.dtype
        assert label.max() <= 20 and label.min() >= 0, 'invalid range'
        return self.colormap[label]

    def show_legend(self):
        fig = plt.figure(figsize=(12, 3), dpi=80, facecolor='w', edgecolor='k')
        for index, (label, color) in enumerate(zip(self.labels, self.colors)):
            patch = np.full((32, 32, 3), color, dtype=np.uint8)
            axis = fig.add_subplot(2, 11, index+1)
            axis.title.set_text(label)
            axis.axis('off')
            plt.imshow(patch)
        plt.show(block=False)

    def semantic_report(self,semantic, limit=3):
        assert semantic.dtype == np.uint8, semantic.dtype
        assert semantic.shape == (512, 512), semantic.shape 
        assert semantic.min() >= 0 and semantic.max() <= 20 , 'invalid range'
        report = ''
        unique, counts = np.unique(semantic, return_counts=True)
        sort_index = np.argsort(np.array(counts)).tolist()
        sort_index.reverse()
        report_count = 0
        for index in sort_index:
            class_id = unique[index]
            count = counts[index]
            percent = count / (512*512) * 100
            if class_id == 0:
                continue
            if percent > 0.5:
                report += '%s:%.0f%% ' % (self.labels[class_id], percent)
                report_count += 1
                if report_count == limit:
                    break
        return report

    def semantic_classwise_distribution(self,batch):
        assert batch.dtype == np.uint8, batch.dtype
        assert batch.shape[1:] == (512, 512), batch.shape
        assert batch.min() >= 0 and batch.max() <= 20
        buf = []
        for i in range(0, batch.shape[0]):
            semantic = batch[i]
            distribution = np.zeros((21), dtype=np.uint8)
            unique, counts = np.unique(semantic, return_counts=True)
            for index, semantic_class in enumerate(unique):
                distribution[semantic_class] = counts[index] / (512.0*512.0) * 100
            buf.append(distribution)
        return np.array(buf, dtype=np.uint8)

voc = Pascal_Voc_Utill()

def get_target(d_class=8):
    img = np.zeros((512, 512), dtype=np.uint8)
    #cv2.putText(img,'ICRA',(30,250), cv2.FONT_HERSHEY_COMPLEX, 6,(d_class*10),16,cv2.LINE_8)
    #cv2.putText(img,'2020',(30,400), cv2.FONT_HERSHEY_COMPLEX, 6,((d_class+1)*10),16,cv2.LINE_8)

    # cv2.rectangle(img,(150,150),(380,380),(d_class*10),-1)
    # cv2.circle(img,(256,256),150,(d_class*10),-1)

    # cv2.circle(img,(256,256),200,((d_class-1)*10),30)
    # cv2.circle(img,(256,256),120,(d_class*10),30)
    # cv2.circle(img,(256,256),50,((d_class+1)*10),-1)

    cv2.putText(img, 'A', (120, 410), cv2.FONT_HERSHEY_SIMPLEX,15, (d_class*10), 45, cv2.LINE_8)
    #_,img = cv2.threshold(img,127,d_class,cv2.THRESH_BINARY)
    #img = np.full((512,512),d_class,dtype=np.uint8)
    target = np.around(img/10).astype(np.uint8)
    return target

class Segmentation_Wrapper():
    def __init__(self,sess,model_name):
        self.sess = sess
        self.model_name = model_name
        self.folder = 'tmp/weights/deeplab/'
        self.supported = ['xception','mobilenetv2','xception_cityscapes','mobilenetv2_cityscapes']
        if not model_name in self.supported:
            print('Supported Models:',list(self.supported))
            raise ValueError('Unsupported Model:'+ model_name)
        self.load_model()
        self.predict(np.zeros((512,512,3),dtype=np.float32))
        print('> Done')
        
    def load_model(self):
        tag = 'deeplab_' + self.model_name
        fn_weight = auto_download(self.folder,tag)
    
        print('> Loading Model [%s] ' % (self.model_name))
        with self.sess.as_default():
            with tf.variable_scope('model'):
                model = Deeplabv3(weights='pascal_voc', backbone=self.model_name, weights_path = fn_weight)
                self.model = model
                
                model_weights = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='model')
                print('> Saving Model')
                tf.compat.v1.train.Saver(model_weights).save(self.sess, 'tmp/')
                print('> Reload Model')
                tf.compat.v1.train.Saver(model_weights).restore(self.sess, 'tmp/')
        
        self.model_in = model.layers[0].input
        self.model_out = model.layers[-1].output
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

    def resize_back(self, img, dsize = None, BGR = False, inter=cv2.INTER_AREA):
        if dsize == None:
            rows, cols, channals = self.src_shape
        max_dim = max(rows, cols)
        resized = cv2.resize(img, (max_dim,max_dim), interpolation=inter)
        img = resized[:rows, :cols, :]
        if BGR:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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
    
    def predict(self, input_var, output='label'):
        """
            This function can accept 4 kinds of input
            1. a string representing the path of the image
            2. a list of strings representing a batch of images
            3. a (512, 512, 3) numpy.ndarray object
            4. a (batch_size, 512, 512, 3) numpy.ndarray object

            This function can output 3 kinds of classification results
            1. label: top probability prediction class id
            2. logits: raw network logits
            3. softmax: logits after softmax
        """
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

        if output == 'logits':
            return self.sess.run(self.model_out, {self.model_in: img_batch})
        elif output == 'softmax':
            return self.sess.run(self.model_softmax, {self.model_in: img_batch})
        elif output == 'label':
            return self.sess.run(self.model_label_out, {self.model_in: img_batch})
        else:
            raise ValueError('output shoud be one of ["logits","softmax","label"]')
