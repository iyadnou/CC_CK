import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    disable_cuda = SetEnvironmentVariable(name='CUDA_VISIBLE_DEVICES', value='-1')
    ntu_pkg_dir = get_package_share_directory('ntu_robotsim_octomap')

    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ntu_pkg_dir, 'launch', 'navigation', 'navigation.launch.py')
        )
    )

    home_dir = os.path.expanduser('~')
    model_absolute_path = os.path.join(home_dir, 'ros2_ws/src/custom_models/best(1).pt')

    # 1. YOLO Node
    yolo_counter_node = Node(
        package='ntu_robotsim_octomap',
        executable='landmark_counter.py',
        name='landmark_counter_node',
        output='screen',
        parameters=[{
            'model_path': model_absolute_path,
            'device': 'cpu',
            'threshold': 0.4, # Lowered slightly to catch signs earlier
            'input_image_topic': '/rgbd_camera/image'
        }],
        prefix='python3 -u'
    )

    # 2. Teleop Node
    teleop_node = Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        name='teleop_node',
        output='screen',
        prefix='xterm -e'
    )

    # 3. Wall Follower (Exploration)
    wall_follower_process = ExecuteProcess(
        cmd=['python3', '-u', os.path.join(home_dir, 'ros2_ws/src/ntu_robotsim_octomap/launch/wall_follower.py')],
        output='screen'
    )

    # --- VISUAL ODOMETRY (SHADOW MODE) ---
    visual_odometry_node = Node(
        package='rtabmap_odom',
        executable='rgbd_odometry',
        name='visual_odometry',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'frame_id': 'base_link',
            'odom_frame_id': 'vo_odom',       # <--- SHADOW MODE: Don't interfere with real odom
            'publish_tf': False,              # <--- SHADOW MODE: Stop fighting Ground Truth!
            'approx_sync': True,
            'approx_sync_max_interval': 0.5,
            'wait_for_transform': 0.5,
            
            # Anti-Crash Optimizations
            'Mem/ImagePreDecimation': '2',    
            'Odom/Strategy': '1',             
            'Odom/GuessMotion': 'true',       
            'Vis/CorType': '1',               
            'Vis/MinInliers': '10',           
            'Vis/MaxFeatures': '150',         
            'Odom/ResetCountdown': '1'
        }],
        remappings=[
            ('rgb/image', '/rgbd_camera/image'),
            ('depth/image', '/rgbd_camera/depth_image'),
            ('rgb/camera_info', '/rgbd_camera/camera_info'),
            ('odom', '/vo_odom')              # Route output to a dummy topic
        ]
    )

    # --- GLUE CAMERA TO ROBOT BASE ---
    camera_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_static_tf',
        # Roll and Yaw are -90 degrees (-1.5708 rad) to fix the Optical Frame rotation!
        arguments=['0.438', '0.0', '0.272', '-1.5708', '0.0', '-1.5708', 'base_link', 'realsense']
    )

    return LaunchDescription([
        disable_cuda,
        navigation_launch,
        visual_odometry_node,
        camera_tf_node,
        yolo_counter_node,
        teleop_node,
        wall_follower_process
    ])