#!/usr/bin/env python3

from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from yolo_msgs.msg import ObjectData
from yolo_msgs.msg import ModelResults

bridge = CvBridge()

class yolo(Node):
    def __init__(self):
        super().__init__('yolo')

        self.model = YOLO('yolov8n.pt')
        self.model_results = ModelResults()

        self.subscription = self.create_subscription(Image, '/webcam_image', self.camera_callback, 10)
        self.subscription

        self.model_pub = self.create_publisher(ModelResults, '/model_results', 1)

    def camera_callback(self, data):
        img = bridge.imgmsg_to_cv2(data, "bgr8")
        results = self.model(img)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                self.object_data = ObjectData()
                b = box.xyxy[0].to('cpu').detach().numpy().copy()	# get box coordinates in (top, left, bottom, right) format
                c = box.cls
                self.object_data.class_name = self.model.names[int(c)]
                self.object_data.top = int(b[0])
                self.object_data.left = int(b[1])
                self.object_data.bottom = int(b[2])
                self.object_data.right = int(b[3])
                self.model_results.model_results.append(self.object_data)

        self.model_pub.publish(self.model_results)
        self.model_results.model_results.clear()

if __name__ == '__main__':
    rclpy.init(args=None)
    yolo = yolo()
    rclpy.spin(yolo)
    rclpy.shutdown()
