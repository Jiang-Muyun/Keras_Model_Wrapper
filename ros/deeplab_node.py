import os
import sys
import time
import getopt
import shutil
import cv2
import signal
import tensorflow as tf
import numpy as np
import roslib
import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from model_wrapper.utils import *
from model_wrapper.segmentation import *

class Deeplab_Node():
    def __init__(self,warpper,input_topic,output_topic):
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.input_compressed = True if self.input_topic.endswith('/compressed') else False
        self.output_compressed = True if self.output_topic.endswith('/compressed') else False
        self.warpper = warpper

        rospy.init_node('deeplab')
        if self.input_compressed:
            self.sub = rospy.Subscriber(self.input_topic, CompressedImage, self.image_callback, queue_size=1)
        else:
            self.sub = rospy.Subscriber(self.input_topic, Image, self.image_callback, queue_size=1)
        
        if self.output_compressed:
            self.pub = rospy.Publisher(self.output_topic,CompressedImage,queue_size=1)
        else:
            self.pub = rospy.Publisher(self.output_topic,Image,queue_size=1)
        print('> Waiting for topic')

    def image_callback(self,ros_data):
        if self.input_compressed:
            np_arr = np.fromstring(ros_data.data, np.uint8)
            img_input = cv2.imdecode(np_arr,1)
        else:
            img_input = bridge.imgmsg_to_cv2(ros_data)
        
        with Tick('prediction'):
            img_resized = self.warpper.resize_keeping_aspect_ratio(img_input)
            semantic = self.warpper.predict(self.warpper.project(img_resized))[0]
            detections = self.warpper.resize_back(voc.get_label_colormap(semantic))
            print(voc.semantic_report(semantic),end='')
            overlap = cv2.addWeighted(detections, 0.5, img_input, 0.5, 20)

        cv2.imshow('overlap',overlap)
        cv2.waitKey(1)

        if self.output_compressed:
            msg = CompressedImage()
            msg.header = ros_data.header
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', detections)[1]).tostring()
            self.pub.publish(msg)
        else:
            msg = Image()
            msg.header = ros_data.header
            msg.data = detections.tostring()
            self.pub.publish(msg)


def shutdownFunction(signalnum, frame):
    print('Exit')
    rospy.signal_shutdown(0)
signal.signal(signal.SIGINT, shutdownFunction)
signal.signal(signal.SIGTERM, shutdownFunction)

if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sess.as_default()
    tf.compat.v1.keras.backend.set_session(sess)

    wrapper = Segmentation_Wrapper(sess,'mobilenetv2')
    # wrapper = Segmentation_Wrapper(sess,'xception')

    input_topic = '/camera/left/image_raw/compressed'
    output_topic = '/deepab/semantic/compressed'
    node = Deeplab_Node(wrapper,input_topic,output_topic)
    rospy.spin()