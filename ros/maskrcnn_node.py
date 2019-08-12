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
from mask_rcnn.rcnn_warpper import *

class Mask_RCNN_Node():
    def __init__(self,input_topic,output_topic):
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.input_compressed = True if self.input_topic.endswith('/compressed') else False
        self.output_compressed = True if self.output_topic.endswith('/compressed') else False

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
        
        with Tick('interference'):
            print(img_input.shape)
            detections = mask_rcnn_predict(img_input)
            overlap = mask_rcnn_plot(detections,img_input)

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
    input_topic = '/camera/left/image_raw/compressed'
    output_topic = '/deepab/semantic/compressed'
    node = Mask_RCNN_Node(input_topic,output_topic)
    rospy.spin()