from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch file cho full autonomy system
    
    Bao gồm:
    - SLAM với loop closure
    - State Machine
    - Path Planning
    - PX4 Controller
    - PX4 MAVLink Bridge
    """
    
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'camera_topic',
            default_value='/camera/image_raw',
            description='Camera image topic'
        ),
        DeclareLaunchArgument(
            'use_slam',
            default_value='true',
            description='Use SLAM node'
        ),
        DeclareLaunchArgument(
            'use_state_machine',
            default_value='true',
            description='Use state machine'
        ),
        DeclareLaunchArgument(
            'use_path_planner',
            default_value='true',
            description='Use path planner'
        ),
        DeclareLaunchArgument(
            'use_px4_controller',
            default_value='true',
            description='Use PX4 controller'
        ),
        DeclareLaunchArgument(
            'use_px4_bridge',
            default_value='true',
            description='Use PX4 MAVLink bridge'
        ),
        DeclareLaunchArgument(
            'px4_connection',
            default_value='udp:127.0.0.1:14540',
            description='PX4 MAVLink connection string'
        ),
        
        # SLAM Node
        Node(
            package='uav_vision',
            executable='slam_node',
            name='slam_node',
            parameters=[{
                'camera_topic': LaunchConfiguration('camera_topic'),
                'map_resolution': 0.1,
                'map_width': 200,
                'map_height': 200,
                'loop_closure_threshold': 0.5,
                'loop_closure_min_distance': 5.0,
            }],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_slam'))
        ),
        
        # State Machine
        Node(
            package='uav_vision',
            executable='state_machine',
            name='state_machine',
            parameters=[{
                'target_altitude': 2.0,
                'takeoff_timeout': 30.0,
                'landing_timeout': 30.0,
                'emergency_landing_altitude': 0.5,
            }],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_state_machine'))
        ),
        
        # Path Planner
        Node(
            package='uav_vision',
            executable='path_planner',
            name='path_planner',
            parameters=[{
                'waypoint_tolerance': 0.5,
                'path_resolution': 0.1,
                'obstacle_inflation': 0.3,
            }],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_path_planner'))
        ),
        
        # PX4 Controller
        Node(
            package='uav_vision',
            executable='px4_controller',
            name='px4_controller',
            parameters=[{
                'max_velocity': 1.0,
                'waypoint_tolerance': 0.5,
                'lookahead_distance': 1.0,
            }],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_px4_controller'))
        ),
        
        # PX4 MAVLink Bridge
        Node(
            package='uav_vision',
            executable='px4_mavlink_bridge',
            name='px4_mavlink_bridge',
            parameters=[{
                'px4_connection': LaunchConfiguration('px4_connection'),
                'vision_odom_topic': '/uav/slam/odometry',
                'send_rate': 30.0,
            }],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_px4_bridge'))
        ),
        
        # Coverage Planner
        Node(
            package='uav_vision',
            executable='coverage_planner_node',
            name='coverage_planner_node',
            output='screen',
        ),
        
        # Evaluation Node
        Node(
            package='uav_vision',
            executable='evaluation_node',
            name='evaluation_node',
            parameters=[{
                'report_interval': 10.0,
            }],
            output='screen',
        ),
        
        # Performance Monitor
        Node(
            package='uav_vision',
            executable='performance_monitor',
            name='performance_monitor',
            parameters=[{
                'update_rate': 1.0,
            }],
            output='screen',
        ),
        
        # Visualization Node
        Node(
            package='uav_vision',
            executable='visualization_node',
            name='visualization_node',
            output='screen',
        ),
    ])

