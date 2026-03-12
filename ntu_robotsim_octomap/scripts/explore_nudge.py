#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class ExploreNudge(Node):

    def __init__(self):
        super().__init__('explore_nudge')

        self.pub = self.create_publisher(Twist, '/cmd_vel_nav', 10)

        # every 20 seconds give robot a small push
        self.timer = self.create_timer(20.0, self.nudge)

        self.get_logger().info("Explore nudge node running")

    def nudge(self):

        msg = Twist()
        msg.linear.x = 0.15

        self.get_logger().info("Nudging robot forward")

        for _ in range(10):
            self.pub.publish(msg)


def main():
    rclpy.init()
    node = ExploreNudge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
