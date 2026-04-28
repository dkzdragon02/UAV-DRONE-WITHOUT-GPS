#!/usr/bin/env python3
"""
mavros_autonomy.launch.py — Launch file cho MAVROS-based autonomous flight

Khởi động toàn bộ stack tự hành (KHÔNG bao gồm PX4 SITL và MAVROS):
  1. px4_param_setup   — Set PX4 params (failsafe, EKF, battery)
  2. dummy_vision_pose — Cung cấp vision pose cho EKF (fallback to 0,0,0)
  3. state_machine     — Quản lý trạng thái bay
  4. px4_controller    — Điều khiển velocity dựa trên state
  5. px4_mavros_offboard_bridge — Forward velocity → MAVROS, auto ARM + OFFBOARD

Sử dụng:
  ros2 launch uav_vision mavros_autonomy.launch.py
  ros2 launch uav_vision mavros_autonomy.launch.py auto_takeoff:=true
  ros2 launch uav_vision mavros_autonomy.launch.py target_altitude:=3.0

Yêu cầu chạy trước:
  1. PX4 SITL: make px4_sitl gazebo-classic_iris_vision
  2. MAVROS: ros2 launch mavros px4.launch fcu_url:='udp://:14540@127.0.0.1:14557'
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, LogInfo
from launch.substitutions import LaunchConfiguration
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
        description='Auto switch to OFFBOARD mode via MAVROS bridge'
    )
    
    max_velocity_arg = DeclareLaunchArgument(
        'max_velocity', default_value='1.0',
        description='Maximum flight velocity (m/s)'
    )

    px4_param_setup_node = Node(
        package='uav_vision',
        executable='px4_param_setup',
        name='px4_param_setup',
        output='screen',
        parameters=[{
            'retry_interval': 2.0,
            'max_retries': 30,
            'auto_shutdown': True,
        }],
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

    return LaunchDescription([
        target_altitude_arg,
        auto_takeoff_arg,
        auto_arm_arg,
        auto_offboard_arg,
        max_velocity_arg,
        
        LogInfo(msg='═══════════════════════════════════════════'),
        LogInfo(msg='  VUAV Autonomy Stack (MAVROS mode)        '),
        LogInfo(msg='═══════════════════════════════════════════'),

        px4_param_setup_node,       # t=0s   — set params immediately
        dummy_vision_node,          # t=3s   — vision pose for EKF
        state_machine_node,         # t=5s   — state management
        px4_controller_node,        # t=5s   — velocity control
        mavros_bridge_node,         # t=8s   — ARM + OFFBOARD + forward velocity
    ])
