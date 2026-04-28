#!/usr/bin/env python3
"""
mavros_custom.launch.py — Robust MAVROS launcher for VUAV

Replaces the fragile XML node.launch approach that fails when
arguments are passed through tmux send-keys. Uses Python launch
API for reliable MAVROS startup with custom plugin denylist.

Usage:
  ros2 launch uav_vision mavros_custom.launch.py
  ros2 launch uav_vision mavros_custom.launch.py fcu_url:=udp://:14540@127.0.0.1:14557
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    uav_vision_share = get_package_share_directory('uav_vision')
    mavros_share = get_package_share_directory('mavros')

    custom_pluginlists = os.path.join(
        uav_vision_share, 'config', 'mavros_pluginlists.yaml'
    )
    px4_config = os.path.join(mavros_share, 'launch', 'px4_config.yaml')

    fcu_url_arg = DeclareLaunchArgument(
        'fcu_url',
        default_value='udp://:14540@127.0.0.1:14557',
        description='FCU connection URL',
    )
    gcs_url_arg = DeclareLaunchArgument(
        'gcs_url',
        default_value='',
        description='GCS bridge URL (empty = disabled)',
    )
    tgt_system_arg = DeclareLaunchArgument(
        'tgt_system',
        default_value='1',
        description='MAVLink target system ID',
    )
    tgt_component_arg = DeclareLaunchArgument(
        'tgt_component',
        default_value='1',
        description='MAVLink target component ID',
    )

    mavros_node = Node(
        package='mavros',
        executable='mavros_node',
        namespace='mavros',
        output='screen',
        respawn=True,
        respawn_delay=5.0,
        parameters=[
            custom_pluginlists,
            px4_config,
            {
                'fcu_url': LaunchConfiguration('fcu_url'),
                'gcs_url': LaunchConfiguration('gcs_url'),
                'tgt_system': LaunchConfiguration('tgt_system'),
                'tgt_component': LaunchConfiguration('tgt_component'),
                'fcu_protocol': 'v2.0',
            },
        ],
    )

    return LaunchDescription([
        fcu_url_arg,
        gcs_url_arg,
        tgt_system_arg,
        tgt_component_arg,
        LogInfo(msg='═══════════════════════════════════════════'),
        LogInfo(msg='  MAVROS Custom Launcher (VUAV)'),
        LogInfo(msg=f'  Plugin denylist: {custom_pluginlists}'),
        LogInfo(msg=f'  PX4 config: {px4_config}'),
        LogInfo(msg='  Respawn: enabled (5s delay)'),
        LogInfo(msg='═══════════════════════════════════════════'),
        mavros_node,
    ])
