#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class WallFollower(Node):
    def __init__(self):
        super().__init__('wall_follower')

        self.pub = self.create_publisher(Twist, '/cmd_vel_nav', 10)

        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            scan_qos
        )

        self.front_dist = float('inf')
        self.right_dist = float('inf')
        self.scan_ok = False

        self.target_wall = 0.55
        self.kp = 0.9
        self.kd = 0.15
        self.prev_error = 0.0

        self.timer = self.create_timer(0.1, self.control)
        self.get_logger().info('Smooth wall follower running')

    def scan_callback(self, msg):
        front_vals = []
        right_neg_vals = []
        right_pos_vals = []

        for i, r in enumerate(msg.ranges):
            if math.isinf(r) or math.isnan(r):
                continue

            angle = msg.angle_min + i * msg.angle_increment

            if -0.35 < angle < 0.35:
                front_vals.append(r)

            # possible right side 1
            if -2.8 < angle < -2.2:
                right_neg_vals.append(r)

            # possible right side 2
            if 2.2 < angle < 2.8:
                right_pos_vals.append(r)

        if front_vals:
            self.front_dist = min(front_vals)

        right_neg = min(right_neg_vals) if right_neg_vals else float('inf')
        right_pos = min(right_pos_vals) if right_pos_vals else float('inf')

        # pick the side that actually looks like a nearby wall
        self.right_dist = min(right_neg, right_pos)

        self.scan_ok = True
        self.get_logger().info(
            f'front={self.front_dist:.2f} right={self.right_dist:.2f}'
        )

    def control(self):
        cmd = Twist()

        if not self.scan_ok:
            self.pub.publish(cmd)
            return

        # wall ahead -> turn left in place
        if self.front_dist < 0.60:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.55
            self.pub.publish(cmd)
            return

        # no right wall seen yet -> move forward with slight right bias
        if self.right_dist == float('inf') or self.right_dist > 1.2:
            cmd.linear.x = 0.10
            cmd.angular.z = -0.12
            self.pub.publish(cmd)
            return

        error = self.target_wall - self.right_dist
        derivative = error - self.prev_error
        turn = self.kp * error + self.kd * derivative
        self.prev_error = error

        # clamp turn so it doesn't spin
        if turn > 0.25:
            turn = 0.25
        elif turn < -0.25:
            turn = -0.25

        cmd.linear.x = 0.14
        cmd.angular.z = turn
        self.pub.publish(cmd)


def main():
    rclpy.init()
    node = WallFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
