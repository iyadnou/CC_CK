#!/usr/bin/env python3
import time
from geometry_msgs.msg import Twist 
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
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
        
        self.declare_parameter('model_path', '')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('threshold', 0.4)
        self.declare_parameter('input_image_topic', '/rgbd_camera/image')

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.device_type = self.get_parameter('device').get_parameter_value().string_value
        self.conf_thresh = self.get_parameter('threshold').get_parameter_value().double_value
        image_topic = self.get_parameter('input_image_topic').get_parameter_value().string_value

        self.get_logger().info(f"Loading YOLO Model from: {model_path}")
        self.model = YOLO(model_path)

        home_dir = os.path.expanduser('~')
        self.db_path = os.path.join(home_dir, 'landmark_database.csv')
        
        if not os.path.exists(self.db_path):
            with open(self.db_path, 'w') as f:
                f.write("Timestamp,Landmark_Class,Object_Count\n")
                
        self.last_recorded_time = {}
        self.record_cooldown = 5.0 

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.param_client = self.create_client(SetParameters, '/controller_server/set_parameters')
        self.current_nav2_speed = 0.22 
        
        # --- CPU SAVER: Frame Skipper ---
        self.frame_count = 0
        
        self.subscription = self.create_subscription(
            Image, image_topic, self.image_callback, qos_profile_sensor_data)
        self.annotated_pub = self.create_publisher(Image, '/yolo/dbg_image', 10)

    def update_nav2_speed(self, speed_val):
        if not self.param_client.service_is_ready():
            return
        req = SetParameters.Request()
        param_val = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(speed_val))
        req.parameters = [Parameter(name='FollowPath.max_vel_x', value=param_val)]
        self.param_client.call_async(req)
        self.current_nav2_speed = speed_val
        self.get_logger().info(f">>> NAV2 SPEED LIMIT RECONFIGURED TO: {speed_val} m/s <<<")

    def image_callback(self, msg):
        # --- CPU SAVER: ONLY PROCESS 1 OUT OF EVERY 3 FRAMES ---
        self.frame_count += 1
        if self.frame_count % 3 != 0:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            results = self.model(cv_image, device=self.device_type, conf=self.conf_thresh, verbose=False)
            
            class_counts = {}
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            current_time = time.time()
            for class_name, count in class_counts.items():
                last_time = self.last_recorded_time.get(class_name, 0.0)
                if current_time - last_time > self.record_cooldown:
                    readable_time = time.strftime('%H:%M:%S', time.localtime(current_time))
                    with open(self.db_path, 'a') as f:
                        f.write(f"{readable_time},{class_name},{count}\n")
                    self.last_recorded_time[class_name] = current_time
                    self.get_logger().info(f"LOGGED TO FILE: {class_name} | Count: {count}")

            if 'stop_sign' in class_counts:
                self.get_logger().warn("STOP SIGN DETECTED! Halting robot.")
                stop_msg = Twist()
                stop_msg.linear.x = 0.0
                stop_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(stop_msg)
                if self.current_nav2_speed != 0.0:
                    self.update_nav2_speed(0.0)
            elif 'slow_sign' in class_counts:
                if self.current_nav2_speed != 0.1: 
                    self.update_nav2_speed(0.1) 
            elif 'fast_sign' in class_counts:
                if self.current_nav2_speed != 0.3: 
                    self.update_nav2_speed(0.3) 

            if class_counts:
                self.get_logger().info(f"Landmarks Detected: {class_counts}")

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