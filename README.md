# NTU_RobotSim OctoMap

A ROS 2 robotics simulation project for 3D mapping and navigation using OctoMap in Gazebo Fortress.

## Overview

This repository provides a complete simulation environment for mobile robot navigation and 3D occupancy mapping in a maze environment. The project integrates ROS 2 Humble, Gazebo Fortress, and OctoMap to create a realistic robotics testing platform for autonomous navigation and spatial mapping.

## Project Structure

```
ntu_robotsim_octomap/
├── config/              # Configuration files for robot and simulation
├── launch/              # Launch files for different simulation components
├── models/              # Robot and environment models (Jetbot, Maze)
├── worlds/              # Gazebo world files
└── octomap2/            # OctoMap server and related packages
    ├── octomap_msgs/    # OctoMap message definitions
    ├── octomap_server2/ # ROS 2 OctoMap server implementation
    ├── pcl_msgs/        # Point Cloud Library message definitions
    └── perception_pcl/  # PCL-ROS integration packages
```

## Building the Project

1. Navigate to your workspace root:

2. Build the workspace:
```bash
colcon build --symlink-install
```

3. Source the workspace:
```bash
source install/setup.bash
```

## Usage

### Quick Start

The simulation requires launching four components in separate terminals. Follow the order below:

### 1. Launch Maze Simulation

This starts the Gazebo Fortress simulator with the maze environment.

```bash
source install/setup.bash
ros2 launch ntu_robotsim_octomap maze.launch.py
```

### 2. Launch Single Robot Simulation

In a **new terminal**, spawn the Jetbot robot in the maze.

```bash
source install/setup.bash
ros2 launch ntu_robotsim_octomap single_robot_sim.launch.py
```

### 3. Launch Ground Truth Odometry and TF Publisher

In a **new terminal**, start the odometry to transform broadcaster for accurate localization.

```bash
source install/setup.bash
ros2 launch odom_to_tf_ros2 odom_to_tf.launch.py
```

### 4. Launch OctoMap Server

In a **new terminal**, start the OctoMap server for 3D mapping.

```bash
source install/setup.bash
ros2 launch octomap_server2 octomap_server_launch.py 
```

### 5. Visualize in RViz2

Launch RViz2 to visualize the OctoMap and robot state:

```bash
rviz2
```

Add the following displays in RViz2:
- **OccupancyGrid** or **MarkerArray** for OctoMap visualization (topic: `/octomap_full` or `/octomap_point_cloud_centers`)
- **RobotModel** for robot visualization
- **TF** for coordinate frame visualization

## Components Description

### Launch Files

- **maze.launch.py**: Initializes the Gazebo simulation environment with the maze world
- **single_robot_sim.launch.py**: Spawns the Jetbot robot with sensors and controllers
- **odom_to_tf.launch.py**: Broadcasts transform tree from ground truth odometry
- **octomap_server_launch.py**: Starts the OctoMap server for 3D occupancy mapping

### Robot Platform

The **Jetbot** robot is equipped with:
- Differential drive for mobility
- Depth camera for 3D perception
- IMU for orientation sensing
- Odometry for position tracking

## Prerequisites

- **OS**: Ubuntu 22.04 (Jammy)
- **ROS 2**: Humble Hawksbill
- **Gazebo**: Fortress
- **Dependencies**:
  - `ros-humble-gazebo-ros-pkgs`
  - `ros-humble-nav2-bringup`
  - `ros-humble-pcl-ros`
  - `ros-humble-pcl-conversions`
  - OctoMap libraries# CC_CK
