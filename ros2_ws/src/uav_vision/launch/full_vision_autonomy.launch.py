#!/usr/bin/env python3
"""
full_vision_autonomy.launch.py — Launch file cho Vision-based autonomous flight

Khởi động toàn bộ stack Vision + Autonomy (KHÔNG bao gồm PX4 SITL và MAVROS):

  Layer 1 — Infra (t=0s):
    1. px4_param_setup      — Set PX4 params (failsafe, EKF, battery)
    2. dummy_vision_pose    — Fallback vision pose (sẽ switch khi real VO sẵn sàng)

  Layer 2 — Vision (t=3s):
    3. vision_node           — Visual Odometry từ camera → /uav/vision/odometry
    4. slam_node             — SLAM + loop closure → /uav/slam/map, /uav/slam/pose

  Layer 3 — Planning (t=5s):
    5. path_planner          — A* path planning → /uav/path_planner/path
    6. obstacle_avoidance_node — Potential field avoidance (optional, LiDAR)

  Layer 4 — Control (t=5s):
    7. state_machine         — Quản lý trạng thái bay
    8. px4_controller        — Velocity control dựa trên state + path

  Layer 5 — Bridge (t=8s):
    9. px4_mavros_offboard_bridge — Forward velocity → MAVROS, auto ARM + OFFBOARD

  Layer 6 — Monitoring (t=3s):
    10. performance_monitor  — CPU, memory, diagnostics

Sử dụng:
  ros2 launch uav_vision full_vision_autonomy.launch.py
  ros2 launch uav_vision full_vision_autonomy.launch.py auto_takeoff:=true
  ros2 launch uav_vision full_vision_autonomy.launch.py enable_slam:=false
  ros2 launch uav_vision full_vision_autonomy.launch.py enable_obstacle_avoidance:=true

Yêu cầu chạy trước:
  1. PX4 SITL: ./scripts/launch_sitl_mavros.sh (hoặc manual)
  2. MAVROS: ros2 launch mavros px4.launch fcu_url:='udp://:14540@127.0.0.1:14557'

  Hoặc dùng launch_sitl_mavros.sh đã tự động setup PX4+MAVROS.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    TimerAction,
    LogInfo,
    GroupAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    target_altitude_arg = DeclareLaunchArgument(
        'target_altitude', default_value='2.0',
        description='Target altitude for takeoff (meters)'
    )

    auto_takeoff_arg = DeclareLaunchArgument(
        'auto_takeoff', default_value='true',
        description='Auto takeoff when PX4 is armed + OFFBOARD'
    )

    auto_arm_arg = DeclareLaunchArgument(
        'auto_arm', default_value='true',
        description='Auto ARM via MAVROS bridge'
    )

    auto_offboard_arg = DeclareLaunchArgument(
        'auto_offboard', default_value='true',
        description='Auto switch to OFFBOARD mode'
    )

    max_velocity_arg = DeclareLaunchArgument(
        'max_velocity', default_value='1.0',
        description='Maximum flight velocity (m/s)'
    )

    enable_vision_arg = DeclareLaunchArgument(
        'enable_vision', default_value='true',
        description='Enable real Visual Odometry (camera required)'
    )

    enable_slam_arg = DeclareLaunchArgument(
        'enable_slam', default_value='true',
        description='Enable SLAM with loop closure'
    )

    enable_path_planning_arg = DeclareLaunchArgument(
        'enable_path_planning', default_value='true',
        description='Enable A* path planning'
    )

    enable_obstacle_avoidance_arg = DeclareLaunchArgument(
        'enable_obstacle_avoidance', default_value='true',
        description='Enable obstacle avoidance (requires LiDAR/depth data)'
    )

    mission_file_arg = DeclareLaunchArgument(
        'mission_file', default_value='',
        description='Path to mission JSON file (GPS waypoints)'
    )

    enable_monitoring_arg = DeclareLaunchArgument(
        'enable_monitoring', default_value='true',
        description='Enable performance monitoring'
    )

    deployment_mode_arg = DeclareLaunchArgument(
        'deployment_mode', default_value='sitl',
        description='Deployment mode: sitl (disable failsafes) or real (enable failsafes)'
    )

    px4_param_setup_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='uav_vision',
                executable='px4_param_setup',
                name='px4_param_setup',
                output='screen',
                parameters=[{
                    'retry_interval': 2.0,
                    'max_retries': 30,
                    'auto_shutdown': True,
                    'deployment_mode': LaunchConfiguration('deployment_mode'),
                }],
            ),
        ],
    )

    dummy_vision_node = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='uav_vision',
                executable='dummy_vision_pose',
                name='dummy_vision_pose',
                output='screen',
                parameters=[{
                    'publish_rate': 30.0,
                    'stale_timeout': 2.0,
                }],
            ),
        ],
    )

    vision_node = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='uav_vision',
                executable='vision_node',
                name='vision_node',
                output='screen',
                condition=IfCondition(LaunchConfiguration('enable_vision')),
                parameters=[{
                    'use_visual_odometry': True,
                    'use_orb_slam': False,
                    'use_vins': False,
                    'camera_width': 640,
                    'camera_height': 480,
                    'camera_fps': 30,
                    'frame_id': 'vision_odom',
                    'child_frame_id': 'base_link',
                    'camera_ready_timeout': 30.0,
                }],
            ),
        ],
    )

    ekf_fusion_node = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='uav_vision',
                executable='ekf_fusion',
                name='ekf_fusion',
                output='screen',
                condition=IfCondition(LaunchConfiguration('enable_vision')),
                parameters=[{
                    'vo_topic': '/uav/vision/odometry',
                    'imu_topic': '/mavros/imu/data',
                    'fused_odom_topic': '/uav/fused_odometry',
                    'fused_pose_topic': '/uav/fused_pose',
                    'publish_rate': 50.0,
                    'vo_position_noise_std': 0.5,
                    'vo_orientation_noise_std': 0.1,
                    'mahalanobis_threshold': 12.59,
                    'mahalanobis_hard_limit': 50.0,
                    'soft_gate_enabled': True,
                    'max_position_innovation': 50.0,
                    'innovation_gate_min': 3.0,
                    'innovation_gate_scale': 3.0,
                    'max_frame_delta': 2.0,
                    'vo_mode': 'incremental',
                }],
            ),
        ],
    )

    slam_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='uav_vision',
                executable='slam_node',
                name='slam_node',
                output='screen',
                condition=IfCondition(LaunchConfiguration('enable_slam')),
                parameters=[{
                    'camera_topic': '/camera/image_raw',
                    'odom_topic': '/uav/slam/odometry',
                    'map_topic': '/uav/slam/map',
                    'pose_topic': '/uav/slam/pose',
                    'loop_closure_topic': '/uav/slam/loop_closure',
                    'map_resolution': 0.1,
                    'map_width': 200,
                    'map_height': 200,
                    'bow_vocabulary_size': 500,
                    'bow_similarity_threshold': 0.3,
                    'bow_min_feature_matches': 15,
                }],
            ),
        ],
    )

    path_planner_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='uav_vision',
                executable='path_planner',
                name='path_planner',
                output='screen',
                condition=IfCondition(LaunchConfiguration('enable_path_planning')),
                parameters=[{
                    'map_topic': '/uav/slam/map',
                    'pose_topic': '/uav/slam/pose',
                    'path_topic': '/uav/path_planner/path',
                    'waypoint_tolerance': 0.5,
                    'path_resolution': 0.1,
                    'obstacle_inflation': 0.3,
                }],
            ),
        ],
    )

    obstacle_avoidance_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='uav_vision',
                executable='obstacle_avoidance_node',
                name='obstacle_avoidance_node',
                output='screen',
                condition=IfCondition(
                    LaunchConfiguration('enable_obstacle_avoidance')
                ),
                parameters=[{
                    'odom_topic': '/uav/vision/odometry',
                    'scan_topic': '/scan',
                    'max_velocity': 1.0,
                    'safety_distance': 0.5,
                    'emergency_distance': 0.3,
                    'max_repulsive_distance': 2.0,
                    'repulsive_gain': 1.5,
                }],
            ),
        ],
    )

    mission_map_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='uav_vision',
                executable='mission_map_node',
                name='mission_map_node',
                output='screen',
                parameters=[{
                    'mission_file': LaunchConfiguration('mission_file'),
                    'auto_start': False,
                    'waypoint_reached_radius': 1.0,
                    'publish_rate': 1.0,
                    'max_speed': LaunchConfiguration('max_velocity'),
                    'geofence_radius': 50.0,
                }],
            ),
        ],
    )

    state_machine_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='uav_vision',
                executable='state_machine',
                name='state_machine',
                output='screen',
                parameters=[{
                    'target_altitude': LaunchConfiguration('target_altitude'),
                    'auto_takeoff': LaunchConfiguration('auto_takeoff'),
                    'takeoff_timeout': 30.0,
                    'landing_timeout': 30.0,
                    'odom_timeout': 3.0,
                    'geofence_radius': 50.0,
                    'geofence_max_altitude': 30.0,
                }],
            ),
        ],
    )

    px4_controller_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='uav_vision',
                executable='px4_controller',
                name='px4_controller',
                output='screen',
                parameters=[{
                    'target_altitude': LaunchConfiguration('target_altitude'),
                    'max_velocity': LaunchConfiguration('max_velocity'),
                    'takeoff_velocity': 0.8,
                    'landing_velocity': 0.3,
                    'waypoint_tolerance': 0.5,
                    'lookahead_distance': 1.0,
                }],
            ),
        ],
    )

    mavros_bridge_node = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='uav_vision',
                executable='px4_mavros_offboard_bridge',
                name='px4_mavros_offboard_bridge',
                output='screen',
                parameters=[{
                    'velocity_input_topic': '/uav/px4_controller/velocity',
                    'mavros_velocity_topic': '/mavros/setpoint_velocity/cmd_vel',
                    'auto_arm': LaunchConfiguration('auto_arm'),
                    'auto_offboard': LaunchConfiguration('auto_offboard'),
                    'setpoint_rate_hz': 20.0,
                }],
            ),
        ],
    )

    performance_monitor_node = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='uav_vision',
                executable='performance_monitor',
                name='performance_monitor',
                output='screen',
                condition=IfCondition(LaunchConfiguration('enable_monitoring')),
                parameters=[{
                    'update_rate': 1.0,
                }],
            ),
        ],
    )

    return LaunchDescription([
        target_altitude_arg,
        auto_takeoff_arg,
        auto_arm_arg,
        auto_offboard_arg,
        max_velocity_arg,
        enable_vision_arg,
        enable_slam_arg,
        enable_path_planning_arg,
        enable_obstacle_avoidance_arg,
        enable_monitoring_arg,
        mission_file_arg,
        deployment_mode_arg,

        LogInfo(msg='═══════════════════════════════════════════════════════'),
        LogInfo(msg='  VUAV Full Vision Autonomy Stack (MAVROS mode)        '),
        LogInfo(msg='  Vision + SLAM + Path Planning + Control + Mission    '),
        LogInfo(msg='═══════════════════════════════════════════════════════'),

        px4_param_setup_node,       # t=5s   — set params after MAVROS connects
        dummy_vision_node,          # t=3s   — vision pose fallback
        vision_node,                # t=3s   — visual odometry
        ekf_fusion_node,            # t=3s   — IMU + VO fusion
        slam_node,                  # t=5s   — SLAM + mapping
        path_planner_node,          # t=5s   — A* path planning
        obstacle_avoidance_node,    # t=5s   — obstacle avoidance
        mission_map_node,           # t=5s   — GPS→NED mission waypoints
        state_machine_node,         # t=5s   — state management
        px4_controller_node,        # t=5s   — velocity control
        mavros_bridge_node,         # t=8s   — ARM + OFFBOARD + forward velocity
        performance_monitor_node,   # t=3s   — system monitoring
    ])
