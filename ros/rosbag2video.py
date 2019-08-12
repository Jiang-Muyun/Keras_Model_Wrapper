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

    def __init__(self,topic, fn_video, node_name='handler'):
        self.topic = topic
        self.frame_index = 0
        self.save_index = 0
        self.src_compressed = True if self.topic.endswith('/compressed') else False
        self.VideoWriter = None
        self.fn_video = fn_video
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')

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
        
        if self.VideoWriter == None:
            height,width,_ = cv_image.shape
            self.VideoWriter = cv2.VideoWriter(self.fn_video,self.fourcc, 30.0, (width,height))
        
        self.VideoWriter.write(cv_image)
        cv2.imshow('cv_image',cv_image)
        cv2.waitKey(1)


if __name__ == '__main__':
    topic = sys.argv[1]
    fn_video = sys.argv[2]
    handle = Rosbag_Handler(topic,fn_video)
    rospy.spin()
    handle.VideoWriter.release()