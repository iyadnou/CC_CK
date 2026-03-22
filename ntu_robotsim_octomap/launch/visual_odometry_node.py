#!/usr/bin/env python3

import math
from typing import Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge

from tf2_ros import TransformBroadcaster


def rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    q = np.empty((4,), dtype=np.float64)
    trace = np.trace(R)

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q[3] = 0.25 * s
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = (R[0, 2] - R[2, 0]) / s
        q[2] = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q[3] = (R[2, 1] - R[1, 2]) / s
        q[0] = 0.25 * s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q[3] = (R[0, 2] - R[2, 0]) / s
        q[0] = (R[0, 1] + R[1, 0]) / s
        q[1] = 0.25 * s
        q[2] = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q[3] = (R[1, 0] - R[0, 1]) / s
        q[0] = (R[0, 2] + R[2, 0]) / s
        q[1] = (R[1, 2] + R[2, 1]) / s
        q[2] = 0.25 * s

    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


class VisualOdometryNode(Node):
    def __init__(self):
        super().__init__('visual_odometry_node')

        self.declare_parameter('image_topic', '/rgbd_camera/image')
        self.declare_parameter('camera_info_topic', '/rgbd_camera/camera_info')
        self.declare_parameter('odom_topic', '/vo/odom')

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.bridge = CvBridge()

        self.create_subscription(Image, image_topic, self.image_callback, qos)
        self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, qos)
        self.odom_pub = self.create_publisher(Odometry, odom_topic, 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        self.K: Optional[np.ndarray] = None
        self.camera_ready = False

        self.prev_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None

        self.orb = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.R_global = np.eye(3)
        self.t_global = np.zeros((3, 1))

        self.get_logger().info('Visual Odometry node started')

    def camera_info_callback(self, msg: CameraInfo):
        self.K = np.array(msg.k).reshape(3, 3)
        self.camera_ready = True

    def image_callback(self, msg: Image):
        if not self.camera_ready:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return

        matches = self.matcher.knnMatch(self.prev_descriptors, descriptors, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < 20:
            self.get_logger().warn('Not enough matches')
            return

        pts1 = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good])
        pts2 = np.float32([keypoints[m.trainIdx].pt for m in good])

        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC)

        if E is None:
            return

        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K)

        self.t_global += self.R_global @ t
        self.R_global = self.R_global @ R

        self.publish_odometry(msg.header.stamp)

        self.prev_gray = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

    def publish_odometry(self, stamp):
        qx, qy, qz, qw = rotation_matrix_to_quaternion(self.R_global)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'

        odom.pose.pose.position.x = float(self.t_global[0])
        odom.pose.pose.position.y = float(self.t_global[1])
        odom.pose.pose.position.z = float(self.t_global[2])

        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        self.odom_pub.publish(odom)

        self.get_logger().info(
            f'VO pose: x={self.t_global[0][0]:.3f}, y={self.t_global[1][0]:.3f}, z={self.t_global[2][0]:.3f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = VisualOdometryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
