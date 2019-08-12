import os
import sys
import time
import getopt
import shutil
import cv2
import numpy as np
import roslib
import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class Recv_Node():
    def __init__(self,src_topic,node_name='Container_Node'):
        self.src_topic = src_topic
        self.src_compressed = True if self.src_topic.endswith('/compressed') else False

        rospy.init_node(node_name)
        if self.src_compressed:
            self.sub = rospy.Subscriber(self.src_topic, CompressedImage, self.image_callback, queue_size=1)
        else:
            self.sub = rospy.Subscriber(self.src_topic, Image, self.image_callback, queue_size=1)
    
    def image_callback(self,ros_data):
        if self.src_compressed:
            np_arr = np.fromstring(ros_data.data, np.uint8)
            cv_image = cv2.imdecode(np_arr,1)
        else:
            cv_image = bridge.imgmsg_to_cv2(ros_data)

        # deal with cv_image
        print(cv_image.shape)
        cv2.imshow('cv_image',cv_image)
        cv2.waitKey(1)

def main(_):
    src_topic = '/camera/left/image_raw/compressed'
    node = Recv_Node(src_topic)
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv[1:])