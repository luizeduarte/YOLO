#!/usr/bin/env python3

import math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from pcl_conversions import pcl_conversions
from sklearn.cluster import KMeans

class Segmentation(Node):

    def __init__(self):
        super().__init__('segmentation_node')

        self.subscription_image = self.create_subscription(Image, '/webcam_image', self.camera_callback, 10)
        self.subscription_yolo = self.create_subscription(ModelResults, '/model_results', self.yolo_callback, 10)

        self.point_cloud_pub = self.create_publisher(PointCloud2, '/point_cloud', 1)

        self.point_cloud = None
        self.center_pointX = 0
        self.center_pointY = 0

    def camera_callback(self, msg):
        height, width, _ = msg.shape
        self.point_cloud = []

        for y in range(height):
            for x in range(width):
                pixel_color = msg[y, x]
                point_2d = [x, y, pixel_color[0], pixel_color[1], pixel_color[2]]  # X, Y, R, G, B
                self.point_cloud.append(point_2d)

        self.point_cloud = np.array(self.point_cloud)

    def yolo_callback(self, msg):
        yolo_result = msg.model_results[0]
        self.center_pointX = (yolo_result.top + yolo_result.left) / 2
        self.center_pointY = (yolo_result.bottom + yolo_result.right) / 2

        cloud_filtered = self.apply_filters()

        EuclideanDistance, distance_x, distance_y = 0, 0, 0
        threshold = 30

        for cluster in self.extract_clusters(cloud_filtered):
            centroid3D = np.mean(cluster, axis=0)[:3]

            pixel_position = self.calculate_pixel_position(centroid3D)

            distance_x = abs(self.center_pointX - pixel_position[0])
            distance_y = abs(self.center_pointY - pixel_position[1])
            EuclideanDistance = math.sqrt(distance_x**2 + distance_y**2)

            if EuclideanDistance < threshold:
                self.publish_segmented_cloud(cluster)

    def apply_filters(self):
        cloud = np.array(pc2.read_points(self.point_cloud, field_names=("x", "y", "z", "rgb"), skip_nans=True))
        
        # Perform voxel grid downsampling filtering
        voxel_size = 0.01
        cloud_filtered = cloud[::int(1/voxel_size)]

        # Perform passthrough filtering to remove points outside a certain range
        z_min, z_max = 0.78, 1.1
        indices = np.where((cloud_filtered[:, 2] >= z_min) & (cloud_filtered[:, 2] <= z_max))
        cloud_filtered = cloud_filtered[indices]

        return cloud_filtered

    def extract_clusters(self, cloud):
        kmeans = KMeans(n_clusters=3, random_state=0)  # Especifique o número desejado de clusters
        kmeans.fit(cloud[:, :3])
        labels = kmeans.labels_

        clusters = []
        for label in np.unique(labels):
            cluster_indices = np.where(labels == label)
            cluster = cloud[cluster_indices]
            clusters.append(cluster)

        return clusters

    def calculate_pixel_position(self, centroid3D):
        camera_matrix = np.array([
            [547.471175, 0.0, 313.045026],
            [0.0, 547.590335, 237.016225],
            [0.0, 0.0, 1.0]
        ])

        pixel_position = np.array([
            int(centroid3D[0] * camera_matrix[0, 0] / centroid3D[2] + camera_matrix[0, 2]),
            int(centroid3D[1] * camera_matrix[1, 1] / centroid3D[2] + camera_matrix[1, 2])
        ])

        return pixel_position

    def publish_segmented_cloud(self, cluster):
        header = pcl_conversions.make_time()
        cloud_msg = pcl_conversions.array_to_pointcloud2(cluster, stamp=header, frame_id='camera_frame')
        self.point_cloud_pub.publish(cloud_msg)


def main():
    rclpy.init()
    node = Segmentation()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
