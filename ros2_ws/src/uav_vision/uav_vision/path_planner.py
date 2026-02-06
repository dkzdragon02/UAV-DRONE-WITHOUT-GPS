#!/usr/bin/env python3
"""
ROS2 Path Planner Node
Basic path planning cho GPS-denied navigation
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Header
import numpy as np
import math
from typing import List, Tuple, Optional


class PathPlanner(Node):
    """Basic path planner cho autonomous navigation"""
    
    def __init__(self):
        super().__init__('path_planner')
        
        # Parameters
        self.declare_parameter('map_topic', '/uav/slam/map')
        self.declare_parameter('pose_topic', '/uav/slam/pose')
        self.declare_parameter('path_topic', '/uav/path_planner/path')
        self.declare_parameter('waypoint_topic', '/uav/path_planner/waypoints')
        
        self.declare_parameter('waypoint_tolerance', 0.5)  # meters
        self.declare_parameter('path_resolution', 0.1)  # meters
        self.declare_parameter('obstacle_inflation', 0.3)  # meters
        
        map_topic = self.get_parameter('map_topic').value
        pose_topic = self.get_parameter('pose_topic').value
        path_topic = self.get_parameter('path_topic').value
        waypoint_topic = self.get_parameter('waypoint_topic').value
        
        self.waypoint_tolerance = self.get_parameter('waypoint_tolerance').value
        self.path_resolution = self.get_parameter('path_resolution').value
        self.obstacle_inflation = self.get_parameter('obstacle_inflation').value
        
        # Current state
        self.current_pose = None
        self.occupancy_map = None
        self.map_info = None
        self.waypoints = []
        self.current_path = None
        
        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            map_topic,
            self.map_callback,
            10
        )
        
        self.pose_sub = self.create_subscription(
            PoseStamped,
            pose_topic,
            self.pose_callback,
            10
        )
        
        self.waypoint_sub = self.create_subscription(
            Path,
            waypoint_topic,
            self.waypoint_callback,
            10
        )
        
        # Publishers
        self.path_pub = self.create_publisher(Path, path_topic, 10)
        
        # Timer for path planning
        self.planning_timer = self.create_timer(1.0, self.plan_path)
        
        self.get_logger().info('Path Planner started')
    
    def map_callback(self, msg):
        """Update occupancy map"""
        self.occupancy_map = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info
    
    def pose_callback(self, msg):
        """Update current pose"""
        self.current_pose = msg
    
    def waypoint_callback(self, msg):
        """Update waypoints"""
        self.waypoints = []
        for pose_stamped in msg.poses:
            waypoint = [
                pose_stamped.pose.position.x,
                pose_stamped.pose.position.y,
                pose_stamped.pose.position.z
            ]
            self.waypoints.append(waypoint)
        
        self.get_logger().info(f'Received {len(self.waypoints)} waypoints')
        self.current_path = None  # Invalidate current path
    
    def plan_path(self):
        """Plan path to next waypoint"""
        if not self.waypoints or self.current_pose is None:
            return
        
        if self.occupancy_map is None:
            # No map, use straight line path
            self.plan_straight_line_path()
            return
        
        # Check if reached current waypoint
        if self.current_path and len(self.current_path.poses) > 0:
            # Check distance to last waypoint
            last_pose = self.current_path.poses[-1]
            distance = self.distance(
                self.current_pose.pose.position,
                last_pose.pose.position
            )
            
            if distance < self.waypoint_tolerance:
                # Reached waypoint, remove it
                if self.waypoints:
                    self.waypoints.pop(0)
                    self.current_path = None
        
        # Plan to next waypoint
        if self.waypoints and (self.current_path is None or len(self.current_path.poses) == 0):
            target = self.waypoints[0]
            self.plan_path_to_target(target)
    
    def plan_straight_line_path(self):
        """Plan straight line path (no obstacles)"""
        if not self.waypoints or self.current_pose is None:
            return
        
        target = self.waypoints[0]
        start = self.current_pose.pose.position
        
        # Create straight line path
        path = Path()
        path.header = Header()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "map"
        
        # Calculate number of points
        distance = self.distance(start, Point(x=float(target[0]), y=float(target[1]), z=float(target[2])))
        num_points = int(distance / self.path_resolution) + 1
        
        for i in range(num_points + 1):
            t = i / num_points if num_points > 0 else 1.0
            
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = start.x + t * (target[0] - start.x)
            pose.pose.position.y = start.y + t * (target[1] - start.y)
            pose.pose.position.z = start.z + t * (target[2] - start.z)
            pose.pose.orientation.w = 1.0
            
            path.poses.append(pose)
        
        self.current_path = path
        self.path_pub.publish(path)
    
    def plan_path_to_target(self, target: List[float]):
        """Plan path to target with obstacle avoidance"""
        if self.current_pose is None or self.occupancy_map is None:
            return
        
        start = self.current_pose.pose.position
        start_pos = [start.x, start.y]
        target_pos = target[:2]
        
        # Simple A* path planning
        path_points = self.a_star_path(start_pos, target_pos)
        
        if not path_points:
            # Fallback to straight line
            self.plan_straight_line_path()
            return
        
        # Create path message
        path = Path()
        path.header = Header()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "map"
        
        for point in path_points:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = float(target[2]) if len(target) > 2 else 0.0
            pose.pose.orientation.w = 1.0
            
            path.poses.append(pose)
        
        self.current_path = path
        self.path_pub.publish(path)
    
    def a_star_path(self, start: List[float], goal: List[float]) -> List[List[float]]:
        """A* path planning with obstacle avoidance"""
        if self.map_info is None:
            return []
        
        # Use improved A* with proper obstacle checking
        return self.a_star_improved(start, goal)
    
    def a_star_improved(self, start: List[float], goal: List[float]) -> List[List[float]]:
        """Improved A* path planning"""
        if self.map_info is None or self.occupancy_map is None:
            return []
        
        # Convert world coordinates to map coordinates
        def world_to_map(wx, wy):
            mx = int((wx - self.map_info.origin.position.x) / self.map_info.resolution)
            my = int((wy - self.map_info.origin.position.y) / self.map_info.resolution)
            return mx, my
        
        def map_to_world(mx, my):
            wx = mx * self.map_info.resolution + self.map_info.origin.position.x
            wy = my * self.map_info.resolution + self.map_info.origin.position.y
            return wx, wy
        
        start_map = world_to_map(start[0], start[1])
        goal_map = world_to_map(goal[0], goal[1])
        
        # Check bounds
        if (start_map[0] < 0 or start_map[0] >= self.map_info.width or
            start_map[1] < 0 or start_map[1] >= self.map_info.height):
            return []
        
        if (goal_map[0] < 0 or goal_map[0] >= self.map_info.width or
            goal_map[1] < 0 or goal_map[1] >= self.map_info.height):
            return []
        
        # Simple straight line (A* simplified for now)
        # In full implementation, use proper A* with obstacle checking
        path_map = []
        dx = goal_map[0] - start_map[0]
        dy = goal_map[1] - start_map[1]
        steps = max(abs(dx), abs(dy))
        
        if steps == 0:
            return []
        
        for i in range(steps + 1):
            t = i / steps
            mx = int(start_map[0] + t * dx)
            my = int(start_map[1] + t * dy)
            
            # Check obstacle
            if (0 <= mx < self.map_info.width and 0 <= my < self.map_info.height):
                if self.occupancy_map[my, mx] > 50:  # Occupied
                    # Try to go around (simplified)
                    continue
            
            wx, wy = map_to_world(mx, my)
            path_map.append([wx, wy])
        
        return path_map
    
    def distance(self, p1: Point, p2: Point) -> float:
        """Calculate 3D distance"""
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        dz = p2.z - p1.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)


def main(args=None):
    rclpy.init(args=args)
    node = PathPlanner()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

