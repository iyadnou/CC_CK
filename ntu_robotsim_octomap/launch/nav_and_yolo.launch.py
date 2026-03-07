import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # Force CPU at the environment level
    disable_cuda = SetEnvironmentVariable(name='CUDA_VISIBLE_DEVICES', value='-1')

    # Get package directory
    ntu_pkg_dir = get_package_share_directory('ntu_robotsim_octomap')

    # Include Navigation
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ntu_pkg_dir, 'launch', 'navigation', 'navigation.launch.py')
        )
    )

    # Safely resolve home directory for the model path
    home_dir = os.path.expanduser('~')
    model_absolute_path = os.path.join(home_dir, 'ros2_ws/custom_models/best.pt')

    # Launch Custom YOLO Counting Node
    yolo_counter_node = Node(
        package='ntu_robotsim_octomap',
        executable='landmark_counter.py',
        name='landmark_counter_node',
        output='screen',
        parameters=[{
            'model_path': model_absolute_path,
            'device': 'cpu',
            'threshold': 0.5,
            'input_image_topic': '/rgbd_camera/image'
        }]
    )

    # Launch teleop for debugging/ moving robot around
    teleop_node = Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        name='teleop_node',
        output='screen',
        prefix='xterm -e'
    )


    return LaunchDescription([
        disable_cuda,
        navigation_launch,
        yolo_counter_node,
	teleop_node
    ])