import os
import sys
import time
import getopt
import shutil
import cv2
import roslib
import rospy
import numpy as np
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()

class Rosbag_Handler():

    def __init__(self,topic,folder,node_name='handler'):
        self.topic = topic
        self.folder = folder
        self.frame_index = 0
        self.save_index = 0
        self.src_compressed = True if self.topic.endswith('/compressed') else False
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        rospy.init_node(node_name)
        if self.src_compressed:
            self.sub = rospy.Subscriber(self.topic, CompressedImage, self.image_callback, queue_size=1)
        else:
            self.sub = rospy.Subscriber(self.topic, Image, self.image_callback, queue_size=1)
    
    def image_callback(self,ros_data):
        if self.src_compressed:
            np_arr = np.fromstring(ros_data.data, np.uint8)
            cv_image = cv2.imdecode(np_arr,1)
        else:
            cv_image = bridge.imgmsg_to_cv2(ros_data)
        
        self.frame_index += 1
        if self.frame_index % 5 == 0:
            self.save_index += 1
            fn = os.path.join(self.folder,'frame_%d.jpg'%(self.save_index))
            cv2.imwrite(fn,cv_image)
            print(fn,self.save_index)
        cv2.imshow('cv_image',cv_image)
        cv2.waitKey(1)

if __name__ == '__main__':
    topic = sys.argv[1]
    folder = sys.argv[2]
    os.makedirs(folder,exist_ok = True)
    handle = Rosbag_Handler(topic,folder)
    rospy.spin()