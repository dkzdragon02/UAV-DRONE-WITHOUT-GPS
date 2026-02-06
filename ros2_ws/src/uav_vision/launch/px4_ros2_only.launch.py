from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    """Launch file cho ROS2-only setup (không cần ROS1 MAVROS)
    
    Sử dụng px4_mavlink_bridge để kết nối trực tiếp với PX4 qua MAVLink
    """
    
    px4_autopilot_dir = os.path.expanduser('~/PX4-Autopilot')
    
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'world',
            default_value='empty',
            description='Gazebo world name'
        ),
        DeclareLaunchArgument(
            'model',
            default_value='iris',
            description='PX4 model'
        ),
        DeclareLaunchArgument(
            'px4_connection',
            default_value='udp:127.0.0.1:14540',
            description='PX4 MAVLink connection string'
        ),
        DeclareLaunchArgument(
            'use_vision',
            default_value='true',
            description='Launch Vision System'
        ),
        DeclareLaunchArgument(
            'use_px4_bridge',
            default_value='true',
            description='Launch PX4 MAVLink bridge'
        ),
        
        # PX4 MAVLink Bridge (ROS2-only, kết nối trực tiếp với PX4)
        Node(
            package='uav_vision',
            executable='px4_mavlink_bridge',
            name='px4_mavlink_bridge',
            parameters=[{
                'px4_connection': LaunchConfiguration('px4_connection'),
                'vision_odom_topic': '/uav/vision/odometry',
                'send_rate': 30.0,
            }],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_px4_bridge'))
        ),
        
        # Vision Node
        Node(
            package='uav_vision',
            executable='vision_node',
            name='vision_node',
            parameters=[{
                'use_visual_odometry': True,
                'use_orb_slam': False,
                'use_vins': False,
                'camera_width': 640,
                'camera_height': 480,
                'camera_fps': 30,
            }],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_vision'))
        ),
        
        # Optical Flow Node
        Node(
            package='uav_vision',
            executable='optical_flow_node',
            name='optical_flow_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_vision'))
        ),
    ])

