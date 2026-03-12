#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction


def generate_launch_description():
    pointcloud_to_scan = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'pointcloud_to_laserscan',
            'pointcloud_to_laserscan_node',
            '--ros-args',
            '-r', 'cloud_in:=/rgbd_camera/points',
            '-r', 'scan:=/scan',
            '-p', 'target_frame:=base_link',
            '-p', 'min_height:=-0.2',
            '-p', 'max_height:=0.5',
            '-p', 'use_sim_time:=true',
        ],
        output='screen'
    )

    explore = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'explore_lite', 'explore',
            '--ros-args',
            '-p', 'costmap_topic:=/global_costmap/costmap',
            '-p', 'map_topic:=/projected_map',
            '-p', 'use_sim_time:=true',
            '-p', 'planner_frequency:=0.3',
            '-p', 'progress_timeout:=60.0',
        ],
        output='screen',
        respawn=True,
        respawn_delay=3.0,
    )

    return LaunchDescription([
        pointcloud_to_scan,
        TimerAction(
            period=8.0,
            actions=[explore]
        )
    ])
