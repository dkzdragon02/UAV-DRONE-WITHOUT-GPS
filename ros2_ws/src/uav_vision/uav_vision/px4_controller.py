#!/usr/bin/env python3
"""
ROS2 PX4 Controller Node
Điều khiển PX4 dựa trên path và state machine commands
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, TwistStamped, Point
from std_msgs.msg import String, Bool, Float64
import numpy as np
import math


class PX4Controller(Node):
    """Controller để điều khiển PX4"""
    
    def __init__(self):
        super().__init__('px4_controller')
        
        # Parameters
        self.declare_parameter('path_topic', '/uav/path_planner/path')
        self.declare_parameter('pose_topic', '/uav/slam/pose')
        self.declare_parameter('state_topic', '/uav/state_machine/state')
        self.declare_parameter('velocity_topic', '/uav/px4_controller/velocity')
        self.declare_parameter('position_target_topic', '/uav/px4_controller/position_target')
        
        self.declare_parameter('max_velocity', 1.0)  # m/s
        self.declare_parameter('waypoint_tolerance', 0.5)  # meters
        self.declare_parameter('lookahead_distance', 1.0)  # meters
        
        path_topic = self.get_parameter('path_topic').value
        pose_topic = self.get_parameter('pose_topic').value
        state_topic = self.get_parameter('state_topic').value
        
        self.max_velocity = self.get_parameter('max_velocity').value
        self.waypoint_tolerance = self.get_parameter('waypoint_tolerance').value
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        
        # Current state
        self.current_pose = None
        self.current_path = None
        self.current_state = "idle"
        self.current_path_index = 0
        
        # Subscribers
        self.path_sub = self.create_subscription(
            Path,
            path_topic,
            self.path_callback,
            10
        )
        
        self.pose_sub = self.create_subscription(
            PoseStamped,
            pose_topic,
            self.pose_callback,
            10
        )
        
        self.state_sub = self.create_subscription(
            String,
            state_topic,
            self.state_callback,
            10
        )
        
        # Publishers
        self.velocity_pub = self.create_publisher(
            TwistStamped,
            self.get_parameter('velocity_topic').value,
            10
        )
        
        self.position_target_pub = self.create_publisher(
            PoseStamped,
            self.get_parameter('position_target_topic').value,
            10
        )
        
        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('PX4 Controller started')
    
    def path_callback(self, msg):
        """Update current path"""
        self.current_path = msg
        self.current_path_index = 0
    
    def pose_callback(self, msg):
        """Update current pose"""
        self.current_pose = msg
    
    def state_callback(self, msg):
        """Update current state"""
        self.current_state = msg.data.lower()
    
    def control_loop(self):
        """Main control loop"""
        if self.current_pose is None:
            return
        
        # State-based control
        if self.current_state == "navigation" and self.current_path:
            self.follow_path()
        elif self.current_state == "hover":
            self.hover_control()
        elif self.current_state == "takeoff":
            self.takeoff_control()
        elif self.current_state == "landing":
            self.landing_control()
        elif self.current_state == "rtl":
            self.rtl_control()
    
    def follow_path(self):
        """Follow planned path"""
        if not self.current_path or len(self.current_path.poses) == 0:
            return
        
        # Find closest point on path
        closest_index = self.find_closest_path_point()
        
        # Find lookahead point
        lookahead_index = self.find_lookahead_point(closest_index)
        
        if lookahead_index >= len(self.current_path.poses):
            lookahead_index = len(self.current_path.poses) - 1
        
        target_pose = self.current_path.poses[lookahead_index]
        
        # Calculate velocity command
        velocity = self.calculate_velocity(target_pose)
        
        # Publish velocity command
        self.publish_velocity(velocity)
    
    def find_closest_path_point(self) -> int:
        """Find closest point on path to current position"""
        if not self.current_path or len(self.current_path.poses) == 0:
            return 0
        
        min_dist = float('inf')
        closest_index = 0
        
        current_pos = self.current_pose.pose.position
        
        for i, path_pose in enumerate(self.current_path.poses):
            dist = self.distance(current_pos, path_pose.pose.position)
            if dist < min_dist:
                min_dist = dist
                closest_index = i
        
        return closest_index
    
    def find_lookahead_point(self, start_index: int) -> int:
        """Find lookahead point on path"""
        if not self.current_path or len(self.current_path.poses) == 0:
            return 0
        
        current_pos = self.current_pose.pose.position
        
        for i in range(start_index, len(self.current_path.poses)):
            path_pose = self.current_path.poses[i]
            dist = self.distance(current_pos, path_pose.pose.position)
            
            if dist >= self.lookahead_distance:
                return i
        
        return len(self.current_path.poses) - 1
    
    def calculate_velocity(self, target_pose: PoseStamped) -> np.ndarray:
        """Calculate velocity command to reach target"""
        if self.current_pose is None:
            return np.array([0.0, 0.0, 0.0])
        
        current_pos = self.current_pose.pose.position
        target_pos = target_pose.pose.position
        
        # Calculate desired velocity
        dx = target_pos.x - current_pos.x
        dy = target_pos.y - current_pos.y
        dz = target_pos.z - current_pos.z
        
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        if distance < self.waypoint_tolerance:
            return np.array([0.0, 0.0, 0.0])
        
        # Normalize and scale
        velocity = np.array([dx, dy, dz]) / distance * self.max_velocity
        
        # Limit velocity
        speed = np.linalg.norm(velocity)
        if speed > self.max_velocity:
            velocity = velocity / speed * self.max_velocity
        
        return velocity
    
    def publish_velocity(self, velocity: np.ndarray):
        """Publish velocity command"""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        
        msg.twist.linear.x = float(velocity[0])
        msg.twist.linear.y = float(velocity[1])
        msg.twist.linear.z = float(velocity[2])
        
        self.velocity_pub.publish(msg)
    
    def hover_control(self):
        """Maintain hover position"""
        # Publish zero velocity
        self.publish_velocity(np.array([0.0, 0.0, 0.0]))
    
    def takeoff_control(self):
        """Control during takeoff"""
        # Publish position target at target altitude
        # In real implementation, use PX4 position setpoint
        pass
    
    def landing_control(self):
        """Control during landing"""
        # Descend slowly
        velocity = np.array([0.0, 0.0, -0.3])  # Descend at 0.3 m/s
        self.publish_velocity(velocity)
    
    def rtl_control(self):
        """Control during return to launch"""
        # Navigate to origin
        target = PoseStamped()
        target.pose.position.x = 0.0
        target.pose.position.y = 0.0
        target.pose.position.z = 2.0  # Target altitude
        
        velocity = self.calculate_velocity(target)
        self.publish_velocity(velocity)
    
    def distance(self, p1: Point, p2: Point) -> float:
        """Calculate 3D distance"""
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        dz = p2.z - p1.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)


def main(args=None):
    rclpy.init(args=args)
    node = PX4Controller()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

