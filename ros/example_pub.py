import sys
import cv2
import os
import time
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()

def main(_):
  def image_callback(ros_data):
    cv_image = bridge.imgmsg_to_cv2(ros_data)
    cv_image = cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)
    msg = bridge.cv2_to_imgmsg(cv_image)
    msg.header = ros_data.header
    pub.publish(msg)

  rospy.init_node("RGB_to_Grey")
  sub = rospy.Subscriber('/mv_26806346/image_raw', Image, image_callback, queue_size=1)
  pub = rospy.Publisher('/cam0/image_raw',Image,queue_size=1)  
  rospy.spin()


if __name__ == "__main__":
  main(sys.argv[1:])