# %matplotlib inline
import os
import cv2
import time
import glob
import json
import numpy as np
import keras
import tensorflow as tf
from keras.models import Model
from keras.applications import *
from tensorflow.python.keras import backend as K

from .common import *

tmp = json.load(open('data/imagenet.json'))
imagenet_classes = [tmp[str(x)] for x in range(1000)]

def process_imagenet_prediction(y):
    """
        Do softmax if input is logits
    """
    if len(y.shape) >= 2 and y.shape[1] == 1000:
        y = y[0]
    assert y.shape == (1000,), y.shape

    if y.min() <= -0.0001 or y.max() >= 1.0001 and np.abs(np.sum(y) - 1.0) > 1e-3:
        e_x = np.exp(y - np.max(y))
        y = e_x / e_x.sum()
    return y


def top_n(prediction, n=1, get_classes=False):
    """ 
        Get top N max probability predictions 
    """
    prediction = process_imagenet_prediction(prediction)
    class_sort = np.argsort(-prediction)
    buf = ''
    classes = []
    for i in range(n):
        class_id = class_sort[i]
        classes.append(class_id)
        accuracy = prediction[class_id]*100
        class_name = imagenet_classes[class_id]
        if len(class_name) >= 30:
            class_name = class_name[:30]
        buf += ('%6.2f%% %3d > %s\n' % (accuracy, class_id, class_name))
    buf = buf[:-1]
    return (classes if get_classes else buf)


class Model_Wrapper():
    def __init__(self, sess, model_name, softmax_check = False):
        self.sess = sess
        self.model_name = model_name
        self.weights_folder = 'tmp/weights/keras_application'
        self.weights = {
            'resnet50': 'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
            'mobilenet_v2_0.35': 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_224.h5',
            'mobilenet_v2_0.5': 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224.h5',
            'mobilenet_v2_0.75': 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_224.h5',
            'mobilenet_v2_1.0': 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5',
            'mobilenet_v2_1.3': 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.3_224.h5',
            'mobilenet_v2_1.4': 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.4_224.h5',
            'xception': 'xception_weights_tf_dim_ordering_tf_kernels.h5',
            'inception_v3': 'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
            'inception_resnet_v2': 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5',
            'vgg16': 'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
            'vgg19': 'vgg19_weights_tf_dim_ordering_tf_kernels.h5',
            'densenet121': 'densenet121_weights_tf_dim_ordering_tf_kernels.h5',
            'densenet169': 'densenet169_weights_tf_dim_ordering_tf_kernels.h5',
            'densenet201': 'densenet201_weights_tf_dim_ordering_tf_kernels.h5',
            'nasnet-mobile': 'NASNet-mobile.h5',
            'nasnet-large': 'NASNet-large.h5'
        }
        if not model_name in self.weights.keys():
            print('Use one of',list(self.weights.keys()))
            raise ValueError('Unsupported Model:'+ model_name)
        self.load_model()
        if softmax_check:
            self.softmax_test()
        print('> Done')

    def softmax_test(self):
        """ 
            Show error message when the softmax layer of the keras model is not removed
        """
        fake_image = np.zeros((224, 224,3),dtype=np.float32)
        y = self.predict(fake_image, 'logits')
        if y.min() >= -0.0001 and y.max() <= 1.0001 and np.allclose(np.sum(y), 1.0, atol=1e-3):
            raise Exception('The softmax layer of the network should be removed.')

    def load_model(self):
        fn_weight = download_file(self.weights_folder,domain+files['classification'][self.model_name])
        print('> Loading Model [%s]' % (self.model_name))

        with self.sess.as_default():
            with tf.variable_scope('model'):
                if self.model_name == 'resnet50':
                    model = ResNet50(input_shape=(224, 224, 3), weights=fn_weight)

                if self.model_name == 'mobilenet_v2_0.35':
                    model = MobileNetV2(input_shape=(224, 224, 3), alpha=0.35, weights=fn_weight)
                if self.model_name == 'mobilenet_v2_0.5':
                    model = MobileNetV2(input_shape=(224, 224, 3), alpha=0.5, weights=fn_weight)
                if self.model_name == 'mobilenet_v2_0.75':
                    model = MobileNetV2(input_shape=(224, 224, 3), alpha=0.75, weights=fn_weight)
                if self.model_name == 'mobilenet_v2_1.0':
                    model = MobileNetV2(input_shape=(224, 224, 3), alpha=1.0, weights=fn_weight)
                if self.model_name == 'mobilenet_v2_1.3':
                    model = MobileNetV2(input_shape=(224, 224, 3), alpha=1.3, weights=fn_weight)
                if self.model_name == 'mobilenet_v2_1.4':
                    model = MobileNetV2(input_shape=(224, 224, 3), alpha=1.4, weights=fn_weight)

                if self.model_name == 'xception':
                    model = Xception(input_shape=(224, 224, 3), weights=fn_weight)
                if self.model_name == 'inception_v3':
                    model = InceptionV3(input_shape=(224, 224, 3), weights=fn_weight)
                if self.model_name == 'inception_resnet_v2':
                    model = InceptionResNetV2(input_shape=(224, 224, 3), weights=fn_weight)

                if self.model_name == 'vgg16':
                    model = VGG16(input_shape=(224, 224, 3), weights=fn_weight)
                if self.model_name == 'vgg19':
                    model = VGG19(input_shape=(224, 224, 3), weights=fn_weight)

                if self.model_name == 'densenet121':
                    model = DenseNet121(input_shape=(224, 224, 3),weights=fn_weight)
                if self.model_name == 'densenet169':
                    model = DenseNet169(input_shape=(224, 224, 3),weights=fn_weight)
                if self.model_name == 'densenet201':
                    model = DenseNet201(input_shape=(224, 224, 3),weights=fn_weight)
                
                if self.model_name == 'nasnet-mobile':
                    model = NASNetMobile(input_shape=(224, 224, 3),weights=fn_weight)
                if self.model_name == 'nasnet-large':
                    model = NASNetLarge(input_shape=(224, 224, 3),weights=fn_weight)
                
            self.model = model
            model_weights = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='model')
            print('> Saving Model')
            tf.compat.v1.train.Saver(model_weights).save(self.sess, 'tmp/')
            print('> Reload Model')
            tf.compat.v1.train.Saver(model_weights).restore(self.sess, 'tmp/')
        
        self.model_in = self.model.layers[0].input
        self.model_out = self.model.layers[-1].output
        self.model_softmax = tf.compat.v1.nn.softmax(self.model_out,-1)
        self.model_label_out = tf.cast(tf.compat.v1.math.argmax(self.model_out,-1),tf.int32)


    def resize_keeping_aspect_ratio(self, img, dsize=(224, 224), inter=cv2.INTER_AREA):
        self.src_shape = img.shape
        rows, cols, channals = img.shape
        max_dim = max(rows, cols)
        tmp = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        tmp[:rows, :cols, :] = img
        resized = cv2.resize(tmp, dsize, interpolation=inter)
        return resized


    def load_image(self, fn, dsize=(224, 224)):
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
            if self.model_name in ['mobilenet_v2', 'xception', 'inception_v3']:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.resize_keeping_aspect_ratio(image, dsize)
            buf.append(image)
        image_batch = np.array(buf, dtype=np.uint8)
        if input_type == 'single':
            return image_batch[0]
        else:
            return image_batch


    def project(self, x):
        """
            Convert Image from [0,255] uint8 to network suitable float32 range 
        """
        if self.model_name in ['resnet50', 'vgg16', 'vgg19']:
            return x.astype(np.float32) - (103.939 + 116.779 + 123.68)/3
        else:
            return (x.astype(np.float32) / 127.5) - 1.0


    def reproject(self, x):
        """
            Convert back float32 Image to [0,255] uint8 for display
        """
        assert x.dtype == np.float32 or x.dtype == np.float64
        if self.model_name in ['resnet50', 'vgg16', 'vgg19']:
            image = (x + (103.939 + 116.779 + 123.68)/3).astype(np.uint8)
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            return ((x + 1.0) * 127.5).astype(np.uint8)


    def predict(self, input_var, output='softmax'):
        """
            This function can accept 4 kinds of input
            1. a string representing the path of the image
            2. a list of strings representing a batch of images
            3. a (224, 224, 3) numpy.ndarray object
            4. a (batch_size, 224, 224, 3) numpy.ndarray object

            This function can output 3 kinds of classification results
            1. Raw network logits (float32)
            2. Logits after softmax (float32)
            3. Top probability prediction class id (int32)

        """
        if isinstance(input_var, str):
            img_batch = self.project(self.load_image([input_var]))
        elif isinstance(input_var, list):
            img_batch = self.project(self.load_image(input_var))
        elif isinstance(input_var, np.ndarray) and input_var.shape == (224, 224, 3):
            img_batch = np.expand_dims(input_var, axis=0)
        elif isinstance(input_var, np.ndarray) and input_var.shape[1:] == (224, 224, 3):
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
