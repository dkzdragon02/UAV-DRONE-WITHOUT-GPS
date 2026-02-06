#!/usr/bin/env python3
"""
ROS2 SLAM Node với Loop Closure Detection
Tích hợp Visual Odometry với map building và loop closure
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, PoseWithCovariance
from std_msgs.msg import Bool, Float64
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque
import time
import math

from uav_vision.visual_odometry import VisualOdometry
from uav_vision.graph_optimizer import GraphOptimizer


class SLAMNode(Node):
    """SLAM Node với loop closure detection"""
    
    def __init__(self):
        super().__init__('slam_node')
        
        # Parameters
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('odom_topic', '/uav/vision/odometry')
        self.declare_parameter('map_topic', '/uav/slam/map')
        self.declare_parameter('pose_topic', '/uav/slam/pose')
        self.declare_parameter('loop_closure_topic', '/uav/slam/loop_closure')
        
        self.declare_parameter('map_resolution', 0.1)  # meters per pixel
        self.declare_parameter('map_width', 200)  # pixels
        self.declare_parameter('map_height', 200)  # pixels
        self.declare_parameter('loop_closure_threshold', 0.5)  # meters
        self.declare_parameter('loop_closure_min_distance', 5.0)  # meters
        self.declare_parameter('loop_closure_history_size', 100)
        
        camera_topic = self.get_parameter('camera_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        map_topic = self.get_parameter('map_topic').value
        pose_topic = self.get_parameter('pose_topic').value
        loop_closure_topic = self.get_parameter('loop_closure_topic').value
        
        # Visual Odometry
        self.visual_odometry = VisualOdometry(
            feature_detector="ORB",
            max_features=500,
            min_match_count=10
        )
        
        # Graph Optimizer
        self.graph_optimizer = GraphOptimizer()
        self.use_graph_optimization = True
        self.optimization_interval = 10  # Optimize every N loop closures
        
        # Map building
        self.map_resolution = self.get_parameter('map_resolution').value
        self.map_width = self.get_parameter('map_width').value
        self.map_height = self.get_parameter('map_height').value
        
        # Initialize occupancy grid map
        self.occupancy_map = np.ones((self.map_height, self.map_width), dtype=np.int8) * -1  # Unknown
        self.map_origin_x = self.map_width // 2
        self.map_origin_y = self.map_height // 2
        
        # Pose history for loop closure
        self.pose_history = deque(maxlen=self.get_parameter('loop_closure_history_size').value)
        self.keyframe_history = deque(maxlen=self.get_parameter('loop_closure_history_size').value)
        
        # Current pose
        self.current_pose = np.array([0.0, 0.0, 0.0])  # x, y, yaw
        self.current_rotation = np.eye(3)
        
        # Loop closure detection
        self.loop_closure_threshold = self.get_parameter('loop_closure_threshold').value
        self.loop_closure_min_distance = self.get_parameter('loop_closure_min_distance').value
        self.last_loop_closure_pose = None
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10
        )
        
        # Publishers
        self.map_pub = self.create_publisher(OccupancyGrid, map_topic, 10)
        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)
        self.odom_pub = self.create_publisher(Odometry, odom_topic, 10)
        self.loop_closure_pub = self.create_publisher(Bool, loop_closure_topic, 10)
        
        # Timer for map publishing
        self.map_timer = self.create_timer(1.0, self.publish_map)
        
        self.get_logger().info('SLAM Node started')
        self.get_logger().info(f'Map resolution: {self.map_resolution}m/pixel')
        self.get_logger().info(f'Map size: {self.map_width}x{self.map_height}')
    
    def image_callback(self, msg):
        """Callback khi nhận image"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Process với visual odometry
            success, translation, rotation, covariance = self.visual_odometry.process_frame(cv_image)
            
            if success:
                # Update pose
                self.update_pose(translation, rotation)
                
                # Check loop closure
                loop_detected = self.detect_loop_closure()
                
                if loop_detected:
                    self.get_logger().info('Loop closure detected!')
                    self.correct_pose_with_loop_closure()
                    
                    # Add to graph optimizer
                    if self.use_graph_optimization:
                        self.add_loop_closure_to_graph()
                
                # Publish pose và odometry
                self.publish_pose(msg.header.stamp)
                self.publish_odometry(msg.header.stamp, translation, rotation, covariance)
                
                # Store keyframe
                self.store_keyframe(cv_image, self.current_pose.copy())
                
                # Update map
                self.update_map()
        
        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')
    
    def update_pose(self, translation: np.ndarray, rotation: np.ndarray):
        """Update current pose từ visual odometry"""
        # Convert rotation matrix to yaw
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        
        # Update position (tích lũy)
        # Transform translation từ camera frame sang world frame
        translation_world = self.current_rotation @ translation
        
        self.current_pose[0] += translation_world[0]
        self.current_pose[1] += translation_world[1]
        self.current_pose[2] = yaw
        
        # Update rotation
        self.current_rotation = self.current_rotation @ rotation
    
    def detect_loop_closure(self) -> bool:
        """Detect loop closure bằng cách so sánh với pose history"""
        if len(self.pose_history) < 10:
            return False
        
        current_pos = self.current_pose[:2]  # x, y only
        
        # Check distance từ last loop closure
        if self.last_loop_closure_pose is not None:
            dist_from_last = np.linalg.norm(current_pos - self.last_loop_closure_pose)
            if dist_from_last < self.loop_closure_min_distance:
                return False  # Too close to last loop closure
        
        # Check với các poses trong history
        for i, past_pose in enumerate(self.pose_history):
            if i < 20:  # Skip recent poses
                continue
            
            past_pos = past_pose[:2]
            distance = np.linalg.norm(current_pos - past_pos)
            
            if distance < self.loop_closure_threshold:
                self.get_logger().info(
                    f'Loop closure candidate: distance={distance:.2f}m, '
                    f'current={current_pos}, past={past_pos}'
                )
                return True
        
        return False
    
    def correct_pose_with_loop_closure(self):
        """Correct pose khi detect loop closure"""
        # Simple correction: average với past pose
        # Trong implementation thực tế, nên dùng graph optimization
        
        current_pos = self.current_pose[:2]
        
        # Find closest past pose
        min_dist = float('inf')
        closest_pose = None
        
        for past_pose in self.pose_history:
            past_pos = past_pose[:2]
            distance = np.linalg.norm(current_pos - past_pos)
            
            if distance < min_dist and distance < self.loop_closure_threshold:
                min_dist = distance
                closest_pose = past_pose
        
        if closest_pose is not None:
            # Simple correction: weighted average
            alpha = 0.3  # Correction weight
            self.current_pose[0] = alpha * closest_pose[0] + (1 - alpha) * self.current_pose[0]
            self.current_pose[1] = alpha * closest_pose[1] + (1 - alpha) * self.current_pose[1]
            
            self.last_loop_closure_pose = self.current_pose[:2].copy()
            
            # Publish loop closure event
            msg = Bool()
            msg.data = True
            self.loop_closure_pub.publish(msg)
    
    def add_loop_closure_to_graph(self):
        """Add loop closure to graph optimizer"""
        if len(self.pose_history) < 2:
            return
        
        current_idx = len(self.pose_history) - 1
        current_pose = self.current_pose
        
        # Find closest past pose
        min_dist = float('inf')
        closest_idx = None
        
        for i, past_pose in enumerate(self.pose_history[:-1]):
            if i < 20:  # Skip recent
                continue
            distance = np.linalg.norm(current_pose[:2] - past_pose[:2])
            if distance < min_dist and distance < self.loop_closure_threshold:
                min_dist = distance
                closest_idx = i
        
        if closest_idx is not None:
            # Add poses to graph if not already added
            pose_id_i = f"pose_{closest_idx}"
            pose_id_j = f"pose_{current_idx}"
            
            if pose_id_i not in self.graph_optimizer.node_indices:
                self.graph_optimizer.add_pose(pose_id_i, self.pose_history[closest_idx])
            if pose_id_j not in self.graph_optimizer.node_indices:
                self.graph_optimizer.add_pose(pose_id_j, current_pose)
            
            # Compute relative pose
            relative_pose = current_pose - self.pose_history[closest_idx]
            
            # Add loop closure constraint
            self.graph_optimizer.add_loop_closure(pose_id_i, pose_id_j, relative_pose)
            
            # Optimize periodically
            if len(self.graph_optimizer.pose_graph.loop_closures) % self.optimization_interval == 0:
                optimized_poses = self.graph_optimizer.optimize(iterations=5)
                # Apply optimized poses
                for pose_id, optimized_pose in optimized_poses.items():
                    idx = int(pose_id.split('_')[1])
                    if idx < len(self.pose_history):
                        self.pose_history[idx] = optimized_pose.copy()
    
    def store_keyframe(self, image: np.ndarray, pose: np.ndarray):
        """Store keyframe cho loop closure"""
        # Store pose
        self.pose_history.append(pose.copy())
        
        # Add to graph optimizer
        if self.use_graph_optimization:
            pose_id = f"pose_{len(self.pose_history) - 1}"
            self.graph_optimizer.add_pose(pose_id, pose)
            
            # Add odometry edge
            if len(self.pose_history) > 1:
                prev_pose_id = f"pose_{len(self.pose_history) - 2}"
                relative_pose = pose - self.pose_history[-2]
                self.graph_optimizer.add_odometry(prev_pose_id, pose_id, relative_pose)
        
        # Store keyframe image (grayscale, resized)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        resized = cv2.resize(gray, (64, 48))  # Small thumbnail
        self.keyframe_history.append(resized)
    
    def update_map(self):
        """Update occupancy grid map"""
        # Convert pose to map coordinates
        map_x = int(self.map_origin_x + self.current_pose[0] / self.map_resolution)
        map_y = int(self.map_origin_y + self.current_pose[1] / self.map_resolution)
        
        # Check bounds
        if 0 <= map_x < self.map_width and 0 <= map_y < self.map_height:
            # Mark current position as free (0)
            # Inflate around position to mark as explored
            radius = 3  # pixels
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    px = map_x + dx
                    py = map_y + dy
                    if 0 <= px < self.map_width and 0 <= py < self.map_height:
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist <= radius:
                            # Free space
                            self.occupancy_map[py, px] = 0
    
    def publish_map(self):
        """Publish occupancy grid map"""
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        
        msg.info.resolution = self.map_resolution
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        msg.info.origin.position.x = -self.map_origin_x * self.map_resolution
        msg.info.origin.position.y = -self.map_origin_y * self.map_resolution
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        
        # Convert numpy array to list
        msg.data = self.occupancy_map.flatten().tolist()
        
        self.map_pub.publish(msg)
    
    def publish_pose(self, stamp):
        """Publish current pose"""
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        
        msg.pose.position.x = float(self.current_pose[0])
        msg.pose.position.y = float(self.current_pose[1])
        msg.pose.position.z = 0.0
        
        # Convert yaw to quaternion
        yaw = self.current_pose[2]
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        
        self.pose_pub.publish(msg)
    
    def publish_odometry(self, stamp, translation, rotation, covariance):
        """Publish odometry"""
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"
        
        # Position
        msg.pose.pose.position.x = float(self.current_pose[0])
        msg.pose.pose.position.y = float(self.current_pose[1])
        msg.pose.pose.position.z = 0.0
        
        # Orientation
        yaw = self.current_pose[2]
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        
        # Covariance (if available)
        if covariance is not None:
            cov_flat = covariance.flatten()
            if len(cov_flat) >= 36:
                msg.pose.covariance = [float(x) for x in cov_flat[:36]]
        
        self.odom_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SLAMNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

