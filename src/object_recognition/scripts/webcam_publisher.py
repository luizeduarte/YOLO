#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class webcam_publisher(Node):
    def __init__(self):
        super().__init__('webcam_publisher')
        self.publisher = self.create_publisher(Image, '/webcam_image', 10)
        self.cv_bridge = CvBridge()

        # start webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3 , 640)
        self.cap.set(4 , 480)
        self.timer = self.create_timer(0.1, self.publish_webcam_image)

    def publish_webcam_image(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert format from BGR to RGB and create the message
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_msg = self.cv_bridge.cv2_to_imgmsg(frame_rgb, encoding="rgb8")

            self.publisher.publish(img_msg)

def main(args=None):
    rclpy.init(args=args)
    node = webcam_publisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
