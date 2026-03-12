import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    # 1. Setup Package Directories
    pkg_octo_map = get_package_share_directory('ntu_robotsim_octomap')
    pkg_nav2_bringup = get_package_share_directory('nav2_bringup')
    pkg_odom_tf = get_package_share_directory('odom_to_tf_ros2')

    # package where Task 1 file lives
    pkg_octomap_server = get_package_share_directory('octomap_server2')

    # Path to parameters
    nav2_params_path = os.path.join(pkg_octo_map, 'config', 'include', 'nav2_octomap_params.yaml')

    # Simulation & Robot
    launch_maze = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_octo_map, 'launch', 'cwmaze.launch.py'))
    )
    launch_robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_octo_map, 'launch', 'single_robot_sim.launch.py'))
    )

    # Odom to TF
    launch_odom_tf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_odom_tf, 'launch', 'odom_to_tf.launch.py'))
    )

    # Filtered OctoMap Server
    launch_octomap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_octomap_server, 'launch', 'octomap_filtered_launch.py'))
    )

    # RViz2 Node
    run_rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen'
    )

    # Nav2 Stack
    launch_nav2 = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(pkg_nav2_bringup, 'launch', 'navigation_launch.py')),
                launch_arguments={
                    'params_file': nav2_params_path,
                    'use_sim_time': 'true',
                    'use_rviz': 'false'  # Prevent conflict with our manual RViz node
                }.items()
            )
        ]
    )

    # Static TF
    static_map_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )




    return LaunchDescription([
        launch_maze,
        launch_robot,
        launch_odom_tf,
        launch_octomap,
        run_rviz,
        launch_nav2,
        static_map_tf,
    ])