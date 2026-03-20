#!/usr/bin/env python3

import math
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.action import ActionClient

from nav_msgs.msg import OccupancyGrid, Odometry
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        self.map_topic = '/projected_map'
        self.odom_topic = '/odom_ground_truth'
        self.scan_topic = '/scan'
        self.cmd_vel_topic = '/cmd_vel'
        self.traffic_stop_topic = '/traffic_stop'

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(OccupancyGrid, self.map_topic, self.map_callback, map_qos)
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, odom_qos)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, scan_qos)
        self.create_subscription(Bool, self.traffic_stop_topic, self.traffic_stop_callback, 10)

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self.traffic_stop = False

        # ===== Map state =====
        self.map_data = None
        self.map_width = 0
        self.map_height = 0
        self.map_res = 0.0
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0

        # ===== Robot pose =====
        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None

        # ===== Current Nav2 goal =====
        self.goal_x = None
        self.goal_y = None
        self.goal_cluster_id = None

        self.goal_active = False
        self.current_goal_handle = None

        # ===== Frontier exploration tuning =====
        self.occ_threshold = 50
        self.safety_radius_cells = 2

        self.min_goal_dist = 1.8
        self.max_goal_dist = 10.0

        self.frontier_min_cluster_size = 20
        self.frontier_search_window_cells = 200

        self.last_log_time = 0.0
        self.have_logged_map = False
        self.have_logged_odom = False
        self.have_logged_scan = False

        self.last_progress_x = None
        self.last_progress_y = None
        self.last_progress_time = time.time()
        self.progress_check_period = 12.0
        self.min_progress_dist = 0.20

        self.rejected_clusters = {}
        self.reject_duration = 45.0

        # Temporary rejection of goals near recent wall-follow exit positions
        self.rejected_frontier_positions = []
        self.reject_position_radius = 1.5
        self.reject_position_duration = 40.0

        # ===== Hybrid supervisor mode =====
        self.mode = 'EXPLORE_WITH_NAV2'
        self.mode_since = time.time()

        # ===== Wall-follow scan state =====
        self.front_dist = float('inf')
        self.front_right_dist = float('inf')
        self.right_dist = float('inf')
        self.scan_ok = False

        # ===== Wall-follow state machine =====
        self.wall_state = 'SEARCH_FOR_WALL'
        self.wall_state_since = time.time()

        # ===== Wall-follow tuning =====
        self.desired_right_dist = 0.55
        self.front_block_enter = 0.50
        self.front_block_exit = 0.68

        self.wall_lost_enter = 0.95
        self.wall_lost_exit = 0.75

        self.linear_follow = 0.12
        self.linear_search = 0.06
        self.angular_turn = 0.42
        self.angular_search = -0.22
        self.max_follow_angular = 0.35

        self.kp = 1.2
        self.kd = 0.18

        self.min_state_time = 0.35
        self.max_range_cap = 3.5

        self.prev_error = 0.0
        self.prev_wall_time = time.time()

        # ===== Hybrid switching thresholds =====
        self.wall_follow_enter_front = 0.45
        self.wall_follow_enter_right = 0.60
        self.wall_follow_exit_front = 1.20
        self.wall_follow_exit_right = 1.40
        self.wall_follow_min_time = 4.0

        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Hybrid frontier explorer started')
        self.get_logger().info(f'Map topic:  {self.map_topic}')
        self.get_logger().info(f'Odom topic: {self.odom_topic}')
        self.get_logger().info(f'Scan topic: {self.scan_topic}')
        self.get_logger().info(f'Cmd topic:  {self.cmd_vel_topic}')
        self.get_logger().info(f'Traffic stop topic: {self.traffic_stop_topic}')
        self.get_logger().info('Using Nav2 action: navigate_to_pose')

    # =========================================================
    # Basic callbacks
    # =========================================================
    def map_callback(self, msg: OccupancyGrid):
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_res = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y
        self.map_data = list(msg.data)

        if not self.have_logged_map:
            self.get_logger().info(
                f'Received map: {self.map_width}x{self.map_height}, res={self.map_res:.3f}'
            )
            self.have_logged_map = True

    def odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

        if not self.have_logged_odom:
            self.get_logger().info(
                f'Received odom: x={self.robot_x:.2f}, y={self.robot_y:.2f}, yaw={self.robot_yaw:.2f}'
            )
            self.have_logged_odom = True

    def scan_callback(self, msg: LaserScan):
        self.front_dist = self.min_in_sector(msg, -0.28, 0.28)
        self.front_right_dist = self.min_in_sector(msg, -0.95, -0.35)
        self.right_dist = self.min_in_sector(msg, -1.75, -1.10)
        self.scan_ok = True

        if not self.have_logged_scan:
            self.get_logger().info(
                f'Received scan: front={self.front_dist:.2f}, '
                f'front_right={self.front_right_dist:.2f}, right={self.right_dist:.2f}'
            )
            self.have_logged_scan = True

    def traffic_stop_callback(self, msg: Bool):
        previous = self.traffic_stop
        self.traffic_stop = msg.data

        if self.traffic_stop and not previous:
            self.get_logger().warn('Traffic stop active -> stopping exploration')
            self.cancel_current_goal()
            self.stop_robot()

        elif (not self.traffic_stop) and previous:
            self.get_logger().info('Traffic stop cleared -> resuming exploration')

    # =========================================================
    # General helpers
    # =========================================================
    def normalize_angle(self, a: float):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def world_to_map(self, x: float, y: float):
        mx = int((x - self.map_origin_x) / self.map_res)
        my = int((y - self.map_origin_y) / self.map_res)
        if 0 <= mx < self.map_width and 0 <= my < self.map_height:
            return mx, my
        return None

    def map_to_world(self, mx: int, my: int):
        x = self.map_origin_x + (mx + 0.5) * self.map_res
        y = self.map_origin_y + (my + 0.5) * self.map_res
        return x, y

    def idx(self, mx: int, my: int):
        return my * self.map_width + mx

    def cell(self, mx: int, my: int):
        return self.map_data[self.idx(mx, my)]

    def is_free(self, mx: int, my: int):
        v = self.cell(mx, my)
        return 0 <= v < self.occ_threshold

    def is_unknown(self, mx: int, my: int):
        return self.cell(mx, my) == -1

    def is_occupied(self, mx: int, my: int):
        return self.cell(mx, my) >= self.occ_threshold

    def neighbors8(self, mx: int, my: int):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = mx + dx
                ny = my + dy
                if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                    yield nx, ny

    # =========================================================
    # Frontier exploration
    # =========================================================
    def is_frontier_cell(self, mx: int, my: int):
        if not self.is_free(mx, my):
            return False
        return any(self.is_unknown(nx, ny) for nx, ny in self.neighbors8(mx, my))

    def is_safe_cell(self, mx: int, my: int):
        for dy in range(-self.safety_radius_cells, self.safety_radius_cells + 1):
            for dx in range(-self.safety_radius_cells, self.safety_radius_cells + 1):
                nx = mx + dx
                ny = my + dy
                if not (0 <= nx < self.map_width and 0 <= ny < self.map_height):
                    return False
                if self.is_occupied(nx, ny):
                    return False
        return True

    def cleanup_rejections(self):
        now = time.time()
        expired = [k for k, t in self.rejected_clusters.items() if t < now]
        for k in expired:
            del self.rejected_clusters[k]

    def cleanup_rejected_positions(self):
        now = time.time()
        self.rejected_frontier_positions = [
            item for item in self.rejected_frontier_positions
            if item['until'] > now
        ]

    def reject_frontiers_near_current_pose(self):
        if self.robot_x is None or self.robot_y is None:
            return

        self.rejected_frontier_positions.append({
            'x': self.robot_x,
            'y': self.robot_y,
            'until': time.time() + self.reject_position_duration,
        })

        self.get_logger().info(
            f'Temporarily rejecting frontier goals near '
            f'({self.robot_x:.2f}, {self.robot_y:.2f})'
        )

    def is_near_rejected_position(self, x, y):
        self.cleanup_rejected_positions()
        for item in self.rejected_frontier_positions:
            if math.hypot(x - item['x'], y - item['y']) < self.reject_position_radius:
                return True
        return False

    def find_frontier_clusters(self):
        if self.map_data is None or self.robot_x is None or self.robot_y is None:
            return []

        robot_cell = self.world_to_map(self.robot_x, self.robot_y)
        if robot_cell is None:
            return []

        rx, ry = robot_cell
        r = self.frontier_search_window_cells
        min_x, max_x = max(0, rx - r), min(self.map_width - 1, rx + r)
        min_y, max_y = max(0, ry - r), min(self.map_height - 1, ry + r)

        frontier_cells = {
            (mx, my)
            for my in range(min_y, max_y + 1)
            for mx in range(min_x, max_x + 1)
            if self.is_frontier_cell(mx, my) and self.is_safe_cell(mx, my)
        }

        visited, clusters = set(), []
        for start in frontier_cells:
            if start in visited:
                continue
            q, cluster = deque([start]), []
            visited.add(start)

            while q:
                cx, cy = q.popleft()
                cluster.append((cx, cy))
                for nx, ny in self.neighbors8(cx, cy):
                    if (nx, ny) in frontier_cells and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        q.append((nx, ny))

            if len(cluster) >= self.frontier_min_cluster_size:
                clusters.append(cluster)

        return clusters

    def cluster_centroid(self, cluster):
        mx = sum(c[0] for c in cluster) / len(cluster)
        my = sum(c[1] for c in cluster) / len(cluster)
        return int(round(mx)), int(round(my))

    def cluster_id(self, cluster):
        return self.cluster_centroid(cluster)

    def choose_goal_from_cluster(self, cluster):
        cx, cy = self.cluster_centroid(cluster)
        for mx, my in sorted(cluster, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2):
            wx, wy = self.map_to_world(mx, my)
            dist = math.hypot(wx - self.robot_x, wy - self.robot_y)
            if self.min_goal_dist <= dist <= self.max_goal_dist:
                return wx, wy
        return None

    def score_cluster(self, cluster):
        cid = self.cluster_id(cluster)
        if cid in self.rejected_clusters:
            return None

        goal = self.choose_goal_from_cluster(cluster)
        if goal is None:
            return None

        wx, wy = goal

        if self.is_near_rejected_position(wx, wy):
            return None

        dist = math.hypot(wx - self.robot_x, wy - self.robot_y)
        yaw_err = abs(self.normalize_angle(
            math.atan2(wy - self.robot_y, wx - self.robot_x) - self.robot_yaw
        ))

        score = (
            1.6 * dist +
            0.5 * yaw_err -
            min(len(cluster), 100) * 0.08
        )
        return score, goal, cid, len(cluster)

    def find_best_frontier_goal(self):
        self.cleanup_rejections()
        self.cleanup_rejected_positions()

        best, best_score = None, float('inf')

        for cluster in self.find_frontier_clusters():
            scored = self.score_cluster(cluster)
            if scored:
                score, goal, cid, size = scored
                if score < best_score:
                    best_score = score
                    best = (goal[0], goal[1], cid, size)

        return best

    # =========================================================
    # Nav2 goal handling
    # =========================================================
    def send_nav_goal(self, x, y):
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('Nav2 action server not available')
            return False

        goal_msg = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation.w = 1.0
        goal_msg.pose = pose

        self.get_logger().info(f'Sending Nav2 goal: ({x:.2f}, {y:.2f})')
        self.nav_client.send_goal_async(goal_msg).add_done_callback(self.goal_response_callback)
        self.goal_active = True
        return True

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected by Nav2')
            self.goal_active = False
            self.current_goal_handle = None
            if self.goal_cluster_id:
                self.rejected_clusters[self.goal_cluster_id] = time.time() + self.reject_duration
            self.goal_x = self.goal_y = self.goal_cluster_id = None
            return

        self.get_logger().info('Goal accepted by Nav2')
        self.current_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        self.get_logger().info('Nav2 goal finished')
        self.goal_active = False
        self.current_goal_handle = None
        self.goal_x = self.goal_y = self.goal_cluster_id = None

    def cancel_current_goal(self):
        if self.current_goal_handle:
            self.get_logger().info('Cancelling current Nav2 goal')
            self.current_goal_handle.cancel_goal_async()
        self.goal_active = False
        self.current_goal_handle = None

    def reject_current_cluster(self):
        if self.goal_cluster_id:
            self.rejected_clusters[self.goal_cluster_id] = time.time() + self.reject_duration
            self.get_logger().info(f'Rejecting frontier cluster {self.goal_cluster_id}')
        self.cancel_current_goal()
        self.goal_x = self.goal_y = self.goal_cluster_id = None

    def check_stuck(self):
        if self.goal_x is None or self.robot_x is None or self.robot_y is None:
            return

        now = time.time()
        if self.last_progress_x is None:
            self.last_progress_x = self.robot_x
            self.last_progress_y = self.robot_y
            self.last_progress_time = now
            return

        moved = math.hypot(self.robot_x - self.last_progress_x, self.robot_y - self.last_progress_y)
        if moved > self.min_progress_dist:
            self.last_progress_x = self.robot_x
            self.last_progress_y = self.robot_y
            self.last_progress_time = now
        elif self.goal_active and now - self.last_progress_time > self.progress_check_period:
            self.get_logger().info('Robot appears stuck; dropping current frontier cluster')
            self.reject_current_cluster()
            self.last_progress_x = self.robot_x
            self.last_progress_y = self.robot_y
            self.last_progress_time = now

    # =========================================================
    # Wall-follow helpers
    # =========================================================
    def set_mode(self, new_mode: str):
        if new_mode != self.mode:
            self.mode = new_mode
            self.mode_since = time.time()
            self.get_logger().info(f'Mode -> {self.mode}')

    def set_wall_state(self, new_state: str):
        if new_state != self.wall_state:
            self.wall_state = new_state
            self.wall_state_since = time.time()
            self.get_logger().info(f'Wall state -> {self.wall_state}')

    def wall_state_elapsed(self) -> float:
        return time.time() - self.wall_state_since

    def sanitize_range(self, r: float) -> float:
        if math.isinf(r) or math.isnan(r):
            return self.max_range_cap
        return min(r, self.max_range_cap)

    def min_in_sector(self, msg: LaserScan, angle_min: float, angle_max: float) -> float:
        vals = []
        angle = msg.angle_min
        for r in msg.ranges:
            if angle_min <= angle <= angle_max:
                vals.append(self.sanitize_range(r))
            angle += msg.angle_increment
        return min(vals) if vals else self.max_range_cap

    def compute_follow_cmd(self) -> Twist:
        cmd = Twist()

        now = time.time()
        dt = max(now - self.prev_wall_time, 1e-3)
        self.prev_wall_time = now

        error = self.desired_right_dist - self.right_dist
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        turn = self.kp * error + self.kd * derivative
        turn = max(min(turn, self.max_follow_angular), -self.max_follow_angular)

        if self.front_right_dist < 0.55:
            cmd.linear.x = 0.07
            cmd.angular.z = max(0.18, turn + 0.08)
        else:
            cmd.linear.x = self.linear_follow
            cmd.angular.z = turn

        return cmd

    def compute_turn_cmd(self) -> Twist:
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = self.angular_turn
        return cmd

    def compute_search_cmd(self) -> Twist:
        cmd = Twist()
        cmd.linear.x = self.linear_search
        cmd.angular.z = self.angular_search
        return cmd

    def decide_wall_state(self):
        front_blocked = self.front_dist < self.front_block_enter
        front_clear = self.front_dist > self.front_block_exit

        wall_lost = self.right_dist > self.wall_lost_enter
        wall_found = self.right_dist < self.wall_lost_exit

        if self.wall_state == 'TURN_LEFT_AT_CORNER':
            if self.wall_state_elapsed() < self.min_state_time:
                return
            if front_clear:
                if wall_found:
                    self.set_wall_state('FOLLOW_WALL')
                else:
                    self.set_wall_state('SEARCH_FOR_WALL')
            return

        if self.wall_state == 'SEARCH_FOR_WALL':
            if self.wall_state_elapsed() < self.min_state_time:
                if front_blocked:
                    self.set_wall_state('TURN_LEFT_AT_CORNER')
                return

            if front_blocked:
                self.set_wall_state('TURN_LEFT_AT_CORNER')
            elif wall_found:
                self.set_wall_state('FOLLOW_WALL')
            return

        if front_blocked:
            self.set_wall_state('TURN_LEFT_AT_CORNER')
        elif wall_lost:
            self.set_wall_state('SEARCH_FOR_WALL')
        else:
            self.set_wall_state('FOLLOW_WALL')

    def run_wall_follow(self):
        cmd = Twist()

        if not self.scan_ok:
            self.cmd_pub.publish(cmd)
            return

        self.decide_wall_state()

        if self.wall_state == 'TURN_LEFT_AT_CORNER':
            cmd = self.compute_turn_cmd()
        elif self.wall_state == 'SEARCH_FOR_WALL':
            cmd = self.compute_search_cmd()
        else:
            cmd = self.compute_follow_cmd()

        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def should_use_wall_follow(self) -> bool:
        if not self.scan_ok:
            return False

        if self.front_dist < self.wall_follow_enter_front:
            return True
        if self.right_dist < self.wall_follow_enter_right:
            return True

        return False

    def should_leave_wall_follow(self) -> bool:
        if not self.scan_ok:
            return False

        if time.time() - self.mode_since < self.wall_follow_min_time:
            return False

        return (
            self.front_dist > self.wall_follow_exit_front and
            self.right_dist > self.wall_follow_exit_right
        )

    # =========================================================
    # Main control loop
    # =========================================================
    def control_loop(self):
        if None in (self.map_data, self.robot_x, self.robot_y, self.robot_yaw):
            now = time.time()
            if now - self.last_log_time > 1.0:
                missing = []
                if self.map_data is None:
                    missing.append(self.map_topic)
                if self.robot_x is None or self.robot_y is None or self.robot_yaw is None:
                    missing.append(self.odom_topic)
                if not self.scan_ok:
                    missing.append(self.scan_topic)
                self.get_logger().info(f'Waiting for: {", ".join(missing)}')
                self.last_log_time = now
            return

        if self.traffic_stop:
            self.cancel_current_goal()
            self.stop_robot()
            return

        # ----- Mode switching -----
        if self.mode == 'EXPLORE_WITH_NAV2' and self.should_use_wall_follow():
            self.get_logger().info('Switching from Nav2 to right-wall-follow')
            self.cancel_current_goal()
            self.set_wall_state('SEARCH_FOR_WALL')
            self.set_mode('FOLLOW_RIGHT_WALL')

        elif self.mode == 'FOLLOW_RIGHT_WALL' and self.should_leave_wall_follow():
            self.get_logger().info('Leaving wall-follow, returning to frontier exploration')
            self.stop_robot()
            self.reject_frontiers_near_current_pose()
            self.set_mode('EXPLORE_WITH_NAV2')
            self.goal_x = None
            self.goal_y = None
            self.goal_cluster_id = None

        # ----- Wall-follow mode -----
        if self.mode == 'FOLLOW_RIGHT_WALL':
            self.run_wall_follow()
            return

        # ----- Nav2 frontier mode -----
        self.check_stuck()

        if self.goal_active:
            now = time.time()
            if now - self.last_log_time > 1.0:
                dist = math.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)
                self.get_logger().info(
                    f'Nav2 goal active: ({self.goal_x:.2f}, {self.goal_y:.2f}) '
                    f'cluster={self.goal_cluster_id} dist={dist:.2f}'
                )
                self.last_log_time = now
            return

        best = self.find_best_frontier_goal()
        if best is None:
            now = time.time()
            if now - self.last_log_time > 2.0:
                self.get_logger().info('No valid frontier cluster found')
                self.last_log_time = now
            return

        self.goal_x, self.goal_y, self.goal_cluster_id, cluster_size = best

        self.get_logger().info(
            f'New frontier cluster goal: ({self.goal_x:.2f}, {self.goal_y:.2f}), '
            f'cluster={self.goal_cluster_id}, size={cluster_size}'
        )

        self.last_progress_x = self.robot_x
        self.last_progress_y = self.robot_y
        self.last_progress_time = time.time()
        self.send_nav_goal(self.goal_x, self.goal_y)


def main(args=None):
    print('Starting hybrid frontier explorer...')
    rclpy.init(args=args)
    node = FrontierExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()