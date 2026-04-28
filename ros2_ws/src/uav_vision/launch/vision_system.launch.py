from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_visual_odometry',
            default_value='true',
            description='Use Visual Odometry'
        ),
        DeclareLaunchArgument(
            'use_orb_slam',
            default_value='false',
            description='Use ORB-SLAM3'
        ),
        DeclareLaunchArgument(
            'use_vins',
            default_value='false',
            description='Use OpenVINS'
        ),
        DeclareLaunchArgument(
            'camera_width',
            default_value='640',
            description='Camera width'
        ),
        DeclareLaunchArgument(
            'camera_height',
            default_value='480',
            description='Camera height'
        ),
        DeclareLaunchArgument(
            'camera_fps',
            default_value='30',
            description='Camera FPS'
        ),
        
        Node(
            package='uav_vision',
            executable='vision_node',
            name='vision_node',
            parameters=[{
                'use_visual_odometry': LaunchConfiguration('use_visual_odometry'),
                'use_orb_slam': LaunchConfiguration('use_orb_slam'),
                'use_vins': LaunchConfiguration('use_vins'),
                'camera_width': LaunchConfiguration('camera_width'),
                'camera_height': LaunchConfiguration('camera_height'),
                'camera_fps': LaunchConfiguration('camera_fps'),
            }],
            output='screen'
        ),
        
        Node(
            package='uav_vision',
            executable='optical_flow_node',
            name='optical_flow_node',
            output='screen'
        ),
        
        Node(
            package='uav_vision',
            executable='px4_mavlink_bridge',
            name='px4_mavlink_bridge',
            parameters=[{
                'px4_connection': 'udp:127.0.0.1:14540',
                'vision_odom_topic': '/uav/vision/odometry',
                'send_rate': 30.0,
            }],
            output='screen'
        ),
    ])

