#!/usr/bin/env python3

import os
import csv
import math
import time

# Force CPU to avoid CUDA initialization errors on incompatible hardware
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import String

from cv_bridge import CvBridge
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from ultralytics import YOLO


def yaw_from_quaternion(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class LandmarkCounterNode(Node):
    def __init__(self):
        super().__init__('landmark_counter_node')

        self.bridge = CvBridge()

        # ===== Parameters =====
        self.declare_parameter('model_path', '')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('threshold', 0.5)
        self.declare_parameter('input_image_topic', '/rgbd_camera/image')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('record_cooldown', 2.0)
        self.declare_parameter('min_record_distance', 0.60)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.device_type = self.get_parameter('device').get_parameter_value().string_value
        self.conf_thresh = self.get_parameter('threshold').get_parameter_value().double_value
        image_topic = self.get_parameter('input_image_topic').get_parameter_value().string_value
        odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.record_cooldown = self.get_parameter('record_cooldown').get_parameter_value().double_value
        self.min_record_distance = self.get_parameter('min_record_distance').get_parameter_value().double_value

        self.get_logger().info(f"Loading YOLO Model from: {model_path}")
        self.get_logger().info(f"Device: {self.device_type} | Threshold: {self.conf_thresh}")

        self.model = YOLO(model_path)

        # ===== CSV database paths =====
        home_dir = os.path.expanduser('~')
        self.db_path = os.path.join(home_dir, 'landmark_database.csv')
        self.summary_path = os.path.join(home_dir, 'landmark_summary.csv')

        self.ensure_csv_headers()

        # ===== Landmark storage =====
        # landmark_memory[class_name] = {
        #   'best_count': int,
        #   'best_x': float,
        #   'best_y': float,
        #   'best_yaw': float,
        #   'last_time': float,
        #   'last_x': float,
        #   'last_y': float,
        # }
        self.landmark_memory = {}

        # ===== Robot pose =====
        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None

        # ===== Traffic-rule state =====
        self.NORMAL_SPEED = 0.22
        self.SLOW_SPEED = 0.10
        self.FAST_SPEED = 0.30

        self.current_mode = "NORMAL"
        self.current_nav2_speed = self.NORMAL_SPEED

        # Stop behavior
        self.stop_active_until = 0.0
        self.stop_hold_time = 3.0  # seconds

        # ===== Publishers / Subscribers / Services =====
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.annotated_pub = self.create_publisher(Image, '/yolo/dbg_image', 10)
        self.detected_pub = self.create_publisher(String, '/detected_landmarks', 10)

        self.param_client = self.create_client(SetParameters, '/controller_server/set_parameters')

        self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            20
        )

        # Timer to continuously enforce stop while stop window is active
        self.stop_timer = self.create_timer(0.1, self.enforce_stop)

    def ensure_csv_headers(self):
        if not os.path.exists(self.db_path):
            with open(self.db_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'landmark_class',
                    'object_count',
                    'robot_x',
                    'robot_y',
                    'robot_yaw_deg'
                ])

        if not os.path.exists(self.summary_path):
            with open(self.summary_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'landmark_class',
                    'best_object_count',
                    'best_robot_x',
                    'best_robot_y',
                    'best_robot_yaw_deg',
                    'last_seen_time'
                ])

    def odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self.robot_x = float(p.x)
        self.robot_y = float(p.y)
        self.robot_yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

    def pose_ready(self) -> bool:
        return self.robot_x is not None and self.robot_y is not None and self.robot_yaw is not None

    def distance_to(self, x1, y1, x2, y2) -> float:
        return math.hypot(x2 - x1, y2 - y1)

    def update_nav2_speed(self, speed_val: float):
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
        now = time.time()

        if now < self.stop_active_until:
            stop_msg = Twist()
            stop_msg.linear.x = 0.0
            stop_msg.angular.z = 0.0
            self.cmd_vel_pub.publish(stop_msg)
        elif self.current_mode == "STOP":
            # Restore normal mode after the hold ends
            self.set_mode("NORMAL")

    def should_record(self, class_name: str, count: int, now: float) -> bool:
        if not self.pose_ready():
            return False

        if class_name not in self.landmark_memory:
            return True

        entry = self.landmark_memory[class_name]
        dt = now - entry['last_time']
        moved = self.distance_to(self.robot_x, self.robot_y, entry['last_x'], entry['last_y'])

        # Record if cooldown passed and robot has moved enough,
        # or if we found a higher count than before.
        if count > entry['best_count']:
            return True

        if dt >= self.record_cooldown and moved >= self.min_record_distance:
            return True

        return False

    def append_detection_to_csv(self, class_name: str, count: int, now: float):
        readable_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))
        yaw_deg = math.degrees(self.robot_yaw)

        with open(self.db_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                readable_time,
                class_name,
                int(count),
                f"{self.robot_x:.3f}",
                f"{self.robot_y:.3f}",
                f"{yaw_deg:.2f}"
            ])

    def update_summary(self):
        with open(self.summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'landmark_class',
                'best_object_count',
                'best_robot_x',
                'best_robot_y',
                'best_robot_yaw_deg',
                'last_seen_time'
            ])

            for class_name in sorted(self.landmark_memory.keys()):
                entry = self.landmark_memory[class_name]
                writer.writerow([
                    class_name,
                    entry['best_count'],
                    f"{entry['best_x']:.3f}",
                    f"{entry['best_y']:.3f}",
                    f"{math.degrees(entry['best_yaw']):.2f}",
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry['last_time']))
                ])

    def store_landmark(self, class_name: str, count: int):
        now = time.time()

        if not self.pose_ready():
            self.get_logger().warn(f"Pose not ready yet, skipping database write for {class_name}")
            return

        if not self.should_record(class_name, count, now):
            return

        if class_name not in self.landmark_memory:
            self.landmark_memory[class_name] = {
                'best_count': int(count),
                'best_x': float(self.robot_x),
                'best_y': float(self.robot_y),
                'best_yaw': float(self.robot_yaw),
                'last_time': now,
                'last_x': float(self.robot_x),
                'last_y': float(self.robot_y),
            }
        else:
            entry = self.landmark_memory[class_name]

            entry['last_time'] = now
            entry['last_x'] = float(self.robot_x)
            entry['last_y'] = float(self.robot_y)

            # Keep the pose associated with the best object count seen so far
            if count >= entry['best_count']:
                entry['best_count'] = int(count)
                entry['best_x'] = float(self.robot_x)
                entry['best_y'] = float(self.robot_y)
                entry['best_yaw'] = float(self.robot_yaw)

        self.append_detection_to_csv(class_name, count, now)
        self.update_summary()

        self.get_logger().info(
            f"LOGGED: {class_name} | count={count} | "
            f"x={self.robot_x:.2f}, y={self.robot_y:.2f}, yaw={math.degrees(self.robot_yaw):.1f}"
        )

    def publish_detected_landmarks(self, class_counts: dict):
        # format: class1:count,class2:count,...
        msg = String()
        items = [f"{k}:{v}" for k, v in sorted(class_counts.items())]
        msg.data = ",".join(items)
        self.detected_pub.publish(msg)

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

            # Store sightings with coordinates and counts
            for class_name, count in class_counts.items():
                self.store_landmark(class_name, count)

            # Publish compact detection info
            if class_counts:
                self.publish_detected_landmarks(class_counts)
                self.get_logger().info(f"Landmarks Detected: {class_counts}")

            # Traffic rule priority: STOP > SLOW > FAST > NORMAL
            if 'stop_sign' in class_counts:
                self.set_mode("STOP")
            elif 'slow_sign' in class_counts:
                self.set_mode("SLOW")
            elif 'fast_sign' in class_counts:
                self.set_mode("FAST")
            else:
                # Only leave STOP after hold time has expired
                if time.time() >= self.stop_active_until:
                    self.set_mode("NORMAL")

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