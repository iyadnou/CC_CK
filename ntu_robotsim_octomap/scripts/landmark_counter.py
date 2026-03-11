#!/usr/bin/env python3
import time
import os

# Force CPU to avoid CUDA initialization errors on incompatible hardware
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue

from ultralytics import YOLO


class LandmarkCounterNode(Node):
    def __init__(self):
        super().__init__('landmark_counter_node')

        self.bridge = CvBridge()

        # Parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('threshold', 0.5)
        self.declare_parameter('input_image_topic', '/rgbd_camera/image')

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.device_type = self.get_parameter('device').get_parameter_value().string_value
        self.conf_thresh = self.get_parameter('threshold').get_parameter_value().double_value
        image_topic = self.get_parameter('input_image_topic').get_parameter_value().string_value

        self.get_logger().info(f"Loading YOLO Model from: {model_path}")
        self.get_logger().info(f"Device: {self.device_type} | Threshold: {self.conf_thresh}")

        self.model = YOLO(model_path)

        # CSV database
        home_dir = os.path.expanduser('~')
        self.db_path = os.path.join(home_dir, 'landmark_database.csv')
        if not os.path.exists(self.db_path):
            with open(self.db_path, 'w') as f:
                f.write("Timestamp,Landmark_Class,Object_Count\n")

        self.last_recorded_time = {}
        self.record_cooldown = 5.0

        # Traffic-rule state
        self.NORMAL_SPEED = 0.22
        self.SLOW_SPEED = 0.10
        self.FAST_SPEED = 0.30

        self.current_mode = "NORMAL"
        self.current_nav2_speed = self.NORMAL_SPEED

        # Stop behavior
        self.stop_active_until = 0.0
        self.stop_hold_time = 3.0  # seconds to hold stop once sign is seen

        # Publishers / Subscribers / Services
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.annotated_pub = self.create_publisher(Image, '/yolo/dbg_image', 10)

        self.param_client = self.create_client(SetParameters, '/controller_server/set_parameters')

        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        # Timer to continuously enforce stop while stop window is active
        self.stop_timer = self.create_timer(0.1, self.enforce_stop)

    def update_nav2_speed(self, speed_val: float):
        """Update both max_vel_x and max_speed_xy for the DWB FollowPath controller."""
        if not self.param_client.service_is_ready():
            self.get_logger().warn("Nav2 parameter service not ready yet.")
            return

        req = SetParameters.Request()
        req.parameters = [
            Parameter(
                name='FollowPath.max_vel_x',
                value=ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE,
                    double_value=float(speed_val)
                )
            ),
            Parameter(
                name='FollowPath.max_speed_xy',
                value=ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE,
                    double_value=float(speed_val)
                )
            )
        ]

        self.param_client.call_async(req)
        self.current_nav2_speed = speed_val
        self.get_logger().info(f"Nav2 speed updated to {speed_val:.2f} m/s")

    def set_mode(self, new_mode: str):
        """Apply traffic rule mode only when it actually changes."""
        if new_mode == self.current_mode:
            return

        self.current_mode = new_mode

        if new_mode == "STOP":
            self.stop_active_until = time.time() + self.stop_hold_time
            self.update_nav2_speed(0.0)
            self.get_logger().warn("STOP sign detected -> robot stopping")

        elif new_mode == "SLOW":
            self.update_nav2_speed(self.SLOW_SPEED)
            self.get_logger().info("SLOW sign detected -> reducing speed")

        elif new_mode == "FAST":
            self.update_nav2_speed(self.FAST_SPEED)
            self.get_logger().info("FAST sign detected -> increasing speed")

        elif new_mode == "NORMAL":
            self.update_nav2_speed(self.NORMAL_SPEED)
            self.get_logger().info("Returning to default speed")

    def enforce_stop(self):
        """Continuously publish zero velocity while stop is active."""
        if time.time() < self.stop_active_until:
            stop_msg = Twist()
            stop_msg.linear.x = 0.0
            stop_msg.angular.z = 0.0
            self.cmd_vel_pub.publish(stop_msg)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            results = self.model(
                cv_image,
                device=self.device_type,
                conf=self.conf_thresh,
                verbose=False
            )

            class_counts = {}
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            # Log detections to CSV with cooldown
            current_time = time.time()
            for class_name, count in class_counts.items():
                last_time = self.last_recorded_time.get(class_name, 0.0)
                if current_time - last_time > self.record_cooldown:
                    readable_time = time.strftime('%H:%M:%S', time.localtime(current_time))
                    with open(self.db_path, 'a') as f:
                        f.write(f"{readable_time},{class_name},{count}\n")
                    self.last_recorded_time[class_name] = current_time
                    self.get_logger().info(f"LOGGED TO FILE: {class_name} | Count: {count}")

            # Traffic rule priority: STOP > SLOW > FAST
            # Change these names if your YOLO classes use different labels.
            if 'stop_sign' in class_counts:
                self.set_mode("STOP")
            elif 'slow_sign' in class_counts:
                self.set_mode("SLOW")
            elif 'fast_sign' in class_counts:
                self.set_mode("FAST")

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