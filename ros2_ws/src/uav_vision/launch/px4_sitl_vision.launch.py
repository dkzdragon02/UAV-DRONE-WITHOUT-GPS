from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
import os

def generate_launch_description():
    px4_autopilot_dir = os.path.expanduser('~/PX4-Autopilot')
    world = LaunchConfiguration('world')
    model = LaunchConfiguration('model')
    use_gazebo = LaunchConfiguration('use_gazebo')
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value='empty',
            description='Gazebo world name (empty, iris, standard_vtol)'
        ),
        DeclareLaunchArgument(
            'model',
            default_value='iris',
            description='PX4 model (iris, standard_vtol, etc.)'
        ),
        DeclareLaunchArgument(
            'use_gazebo',
            default_value='true',
            description='Launch Gazebo simulator'
        ),
        DeclareLaunchArgument(
            'px4_connection',
            default_value='udp:127.0.0.1:18570',
            description='PX4 MAVLink connection string'
        ),
        DeclareLaunchArgument(
            'use_vision',
            default_value='true',
            description='Launch Vision System'
        ),
        DeclareLaunchArgument(
            'use_mock_lidar',
            default_value='true',
            description='Use mock lidar publisher (when real lidar is not available)'
        ),
        
        ExecuteProcess(
            cmd=[
                'bash', '-c',
                [
                    TextSubstitution(text='cd '),
                    TextSubstitution(text=px4_autopilot_dir),
                    TextSubstitution(text=' && ./Tools/simulation/gazebo-classic/sitl_multiple_run.sh -n 1 -m '),
                    model,
                    TextSubstitution(text=' -w '),
                    world
                ]
            ],
            output='screen',
            condition=IfCondition(use_gazebo)
        ),
        
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
            condition=IfCondition(LaunchConfiguration('use_vision'))
        ),
        
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
        
        Node(
            package='uav_vision',
            executable='optical_flow_node',
            name='optical_flow_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_vision'))
        ),
        
        Node(
            package='uav_vision',
            executable='obstacle_avoidance_node',
            name='obstacle_avoidance_node',
            parameters=[{
                'odom_topic': '/uav/vision/odometry',
                'scan_topic': '/scan',
                'goal_x': 5.0,
                'goal_y': 0.0,
                'max_velocity': 1.0,
            }],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_vision'))
        ),
        
        Node(
            package='uav_vision',
            executable='mock_lidar_publisher',
            name='mock_lidar_publisher',
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_mock_lidar'))
        ),
        
    ])

