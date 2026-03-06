
## Building the Project

## 1. Navigate to your workspace root:
cd ros2_ws
## 2. Build
colcon build --symlink-install
## 3. Source:
source install/setup.bash
## 4. Launch
ros2 launch ntu_robotsim_octomap navigation.launch.py

