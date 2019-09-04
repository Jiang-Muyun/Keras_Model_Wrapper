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

sys.path.append('.')
from utils import Tick
from mask_rcnn.warpper import predict,plot

class Mask_RCNN_Node():
    def __init__(self,input_topic):
        self.input_topic = input_topic
        self.input_compressed = True if self.input_topic.endswith('/compressed') else False

        rospy.init_node('mask_rcnn')
        if self.input_compressed:
            self.sub = rospy.Subscriber(self.input_topic, CompressedImage, self.image_callback, queue_size=1)
        else:
            self.sub = rospy.Subscriber(self.input_topic, Image, self.image_callback, queue_size=1)
        print('> Waiting for topic')

    def image_callback(self,ros_data):
        if self.input_compressed:
            np_arr = np.fromstring(ros_data.data, np.uint8)
            img_input = cv2.imdecode(np_arr,1)
        else:
            img_input = bridge.imgmsg_to_cv2(ros_data)
        
        with Tick('interference'):
            detections = predict(img_input)
            overlap = plot(detections,img_input)
            detections = overlap

        cv2.imshow('overlap',overlap)
        cv2.waitKey(1)

def shutdownFunction(signalnum, frame):
    print('Exit')
    rospy.signal_shutdown(0)
signal.signal(signal.SIGINT, shutdownFunction)
signal.signal(signal.SIGTERM, shutdownFunction)

if __name__ == '__main__':
    input_topic = sys.argv[1]
    node = Mask_RCNN_Node(input_topic)
    rospy.spin()