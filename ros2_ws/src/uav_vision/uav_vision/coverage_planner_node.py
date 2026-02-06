#!/usr/bin/env python3
"""
ROS2 Coverage Planner Node
Tạo coverage patterns cho mapping missions
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from uav_vision.coverage_planner import CoveragePlanner


class CoveragePlannerNode(Node):
    """Coverage planner node"""
    
    def __init__(self):
        super().__init__('coverage_planner_node')
        
        # Coverage planner
        self.coverage_planner = CoveragePlanner()
        
        # Parameters
        self.declare_parameter('waypoint_topic', '/uav/path_planner/waypoints')
        self.declare_parameter('pattern_topic', '/uav/coverage_planner/pattern')
        
        waypoint_topic = self.get_parameter('waypoint_topic').value
        pattern_topic = self.get_parameter('pattern_topic').value
        
        # Subscribers
        self.pattern_sub = self.create_subscription(
            String,
            pattern_topic,
            self.pattern_callback,
            10
        )
        
        # Publishers
        self.waypoint_pub = self.create_publisher(Path, waypoint_topic, 10)
        
        self.get_logger().info('Coverage Planner Node started')
        self.get_logger().info('Available patterns: lawnmower, spiral, zigzag, rectangle, circle')
    
    def pattern_callback(self, msg):
        """Handle pattern request"""
        pattern = msg.data.lower()
        
        # Parse pattern (format: "pattern_name:param1:param2:...")
        parts = pattern.split(':')
        pattern_name = parts[0]
        
        try:
            if pattern_name == 'lawnmower':
                # Format: lawnmower:x_min:y_min:x_max:y_max:altitude:spacing
                if len(parts) >= 7:
                    bounds = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                    altitude = float(parts[5])
                    spacing = float(parts[6])
                    waypoints = self.coverage_planner.plan_lawnmower(bounds, altitude, spacing)
                    self.publish_waypoints(waypoints)
            
            elif pattern_name == 'spiral':
                # Format: spiral:cx:cy:max_radius:altitude:spacing
                if len(parts) >= 6:
                    center = (float(parts[1]), float(parts[2]))
                    max_radius = float(parts[3])
                    altitude = float(parts[4])
                    spacing = float(parts[5])
                    waypoints = self.coverage_planner.plan_spiral(center, max_radius, altitude, spacing)
                    self.publish_waypoints(waypoints)
            
            elif pattern_name == 'rectangle':
                # Format: rectangle:cx:cy:width:height:altitude:spacing
                if len(parts) >= 7:
                    center = (float(parts[1]), float(parts[2]))
                    width = float(parts[3])
                    height = float(parts[4])
                    altitude = float(parts[5])
                    spacing = float(parts[6])
                    waypoints = self.coverage_planner.plan_rectangle(center, width, height, altitude, spacing)
                    self.publish_waypoints(waypoints)
            
            elif pattern_name == 'circle':
                # Format: circle:cx:cy:radius:altitude:spacing
                if len(parts) >= 6:
                    center = (float(parts[1]), float(parts[2]))
                    radius = float(parts[3])
                    altitude = float(parts[4])
                    spacing = float(parts[5])
                    waypoints = self.coverage_planner.plan_circle(center, radius, altitude, spacing)
                    self.publish_waypoints(waypoints)
            
            else:
                self.get_logger().warn(f'Unknown pattern: {pattern_name}')
        
        except (ValueError, IndexError) as e:
            self.get_logger().error(f'Error parsing pattern: {e}')
    
    def publish_waypoints(self, waypoints):
        """Publish waypoints as Path"""
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "map"
        
        for wp in waypoints:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(wp[0])
            pose.pose.position.y = float(wp[1])
            pose.pose.position.z = float(wp[2])
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        
        self.waypoint_pub.publish(path)
        self.get_logger().info(f'Published {len(waypoints)} waypoints')


def main(args=None):
    rclpy.init(args=args)
    node = CoveragePlannerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

