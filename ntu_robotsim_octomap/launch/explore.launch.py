from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    return LaunchDescription([
        ExecuteProcess(
            cmd=[
                'bash',
                '-lc',
                'source /home/ntu-user/ros2_ws/install/setup.bash && '
                'ros2 run explore_lite explore --ros-args '
                '--params-file /home/ntu-user/ros2_ws/src/explore/config/params_costmap.yaml '
                '-p use_sim_time:=true'
            ],
            output='screen'
        )
    ])
