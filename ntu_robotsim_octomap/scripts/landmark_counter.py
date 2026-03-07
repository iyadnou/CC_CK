#!/usr/bin/env python3

import os
# Force CPU to avoid CUDA initialization errors on incompatible hardware
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
import cv2
from ultralytics import YOLO

class LandmarkCounterNode(Node):
    def __init__(self):
        super().__init__('landmark_counter_node')
        self.bridge = CvBridge()
        
        # 1. Declare Parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('threshold', 0.5)
        self.declare_parameter('input_image_topic', '/rgbd_camera/image')

        # 2. Retrieve Parameters
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.device_type = self.get_parameter('device').get_parameter_value().string_value
        self.conf_thresh = self.get_parameter('threshold').get_parameter_value().double_value
        image_topic = self.get_parameter('input_image_topic').get_parameter_value().string_value

        self.get_logger().info(f"Loading YOLO Model from: {model_path}")
        self.get_logger().info(f"Device: {self.device_type} | Threshold: {self.conf_thresh}")
        
        # 3. Load Model
        self.model = YOLO(model_path) 
        
        # 4. Subscribe to Camera (Using sensor_data QoS for Gazebo)
        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile_sensor_data)
            
        # 5. Publisher for RViz visualization
        self.annotated_pub = self.create_publisher(Image, '/yolo/dbg_image', 10)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Run Inference (verbose=False stops it from spamming the terminal with ultralytics logs)
            results = self.model(cv_image, device=self.device_type, conf=self.conf_thresh, verbose=False)
            
            # --- COURSEWORK 2.1 REQUIREMENT: COMPLEX RECOGNITION (COUNTING) ---
            class_counts = {}
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Log the counts to the terminal
            if class_counts:
                self.get_logger().info(f"Landmarks Detected: {class_counts}")

            # Publish image to view in RViz
            annotated_frame = results[0].plot()
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            self.annotated_pub.publish(annotated_msg)
            
        except Exception as e:
            self.get_logger().error(f"Image processing failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = LandmarkCounterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()