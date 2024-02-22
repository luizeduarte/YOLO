#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from yolo_msgs.msg import ObjectData
from yolo_msgs.msg import ModelResults
import cv2
import numpy as np
from sklearn.cluster import KMeans
from cv_bridge import CvBridge
from pyntcloud import PyntCloud
import open3d as o3d

bridge = CvBridge()

class cloud_clustering_node(Node):
   def __init__(self):
        super().__init__('cloud_clustering_node')
        self.subscription = self.create_subscription(ModelResults, '/model_results', self.yolo_callback, 10)
        self.subscription = self.create_subscription(Image, '/webcam_image', self.camera_callback, 10)
        self.image_publisher = self.create_publisher(Image, '/segmented_image', 10)
        self.subscription

   def camera_callback(self, data):
        global img
        img = bridge.imgmsg_to_cv2(data, "bgr8")
        self.gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        #use the ORB detector to find key points
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(self.gray_image, None)

        # Converta os keypoints para o formato correto (pontos 2D homogêneos)
        keypoints_homogeneous = np.array([keypoint.pt + (1,) for keypoint in keypoints], dtype=np.float32)

        # Matriz de projeção da câmera (substitua pela sua matriz real)
        projection_matrix = np.eye(3, 4)

        # Triangulação dos pontos
        # Triangulação dos pontos
        points_4d_homogeneous = cv2.triangulatePoints(
                np.eye(3),  # Matriz de projeção da primeira câmera
                projection_matrix,  # Matriz de projeção da segunda câmera
                keypoints_homogeneous.T[:, :2],  # Use apenas as duas primeiras coordenadas dos pontos de projeção
                keypoints_homogeneous.T[:, 2:],  # Use apenas as duas últimas coordenadas dos pontos de projeção
        )

        # Converta para coordenadas 3D não homogêneas
        points_3d = cv2.convertPointsFromHomogeneous(points_4d_homogeneous.T).reshape(-1, 3)

        # Crie uma nuvem de pontos PyntCloud
        self.point_cloud = PyntCloud(points=points_3d)

        # Visualize a nuvem de pontos com Open3D
        o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_3d))])


   def yolo_callback(self, data):

	# get the first object detected by YOLO
        self.yolo_box = ObjectData()
        self.yolo_box = data.model_results[0]


        # extract the 2D points from the YOLO bounding box
        yolo_points_2d = np.array([(self.yolo_box.top + self.yolo_box.left) / 2, (self.yolo_box.bottom + self.yolo_box.right) / 2])


        # combine the 3D points from the point cloud and the 2D points from YOLO
        combined_points = np.hstack((self.point_cloud, yolo_points_2d))


        # K-means clustering
        num_clusters = 10  # Ajust the number of clusters as needed
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        labels = kmeans.fit_predict(combined_points)


        # calculate the centroids of the clusters
        # centroids_3d = np.array([np.mean(self.point_cloud[labels == i], axis=0) for i in range(num_clusters)])
        centroids_2d = np.array([np.mean(yolo_points_2d[labels == i], axis=0) for i in range(num_clusters)])

        # create a mask for the segmented image
        mask = np.zeros_like(self.gray_image)
        mask = cv2.fillPoly(mask, np.int32([centroids_2d]), (255, 255, 255))

        # apply the mask to the original image
        self.segmented_image = cv2.bitwise_and(self.gray_image, self.gray_image, mask=mask)

        # Publish the segmented image
        self.publish_segmented_image(self.segmented_image)
        # save image for testing purposes
        cv2.imwrite('segmentation_result.jpg', img)


   def publish_segmented_image(self, image):
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'  # Adjust the frame_id accordingly
        msg.height, msg.width, msg.step = image.shape
        msg.encoding = 'bgr8'
        msg.data = image.tobytes()
        self.image_publisher.publish(msg)




def main(args=None):
        rclpy.init(args=args)
        node = cloud_clustering_node()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()




if __name__ == '__main__':
        main()
