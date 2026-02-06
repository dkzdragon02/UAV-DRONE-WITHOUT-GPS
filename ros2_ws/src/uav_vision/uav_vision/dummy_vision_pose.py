#!/usr/bin/env python3
"""
Dummy Vision Pose Publisher
Publish dummy vision pose để PX4 EKF có data (giải quyết "ekf2 missing data")
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import time


class DummyVisionPose(Node):
    """Publish dummy vision pose để PX4 EKF có data"""
    
    def __init__(self):
        super().__init__('dummy_vision_pose')
        
        # Parameters
        self.declare_parameter('mavros_vision_topic', '/mavros/vision_pose/pose')
        self.declare_parameter('vision_odom_topic', '/uav/vision/odometry')
        self.declare_parameter('publish_rate', 30.0)  # Hz
        
        mavros_topic = self.get_parameter('mavros_vision_topic').value
        odom_topic = self.get_parameter('vision_odom_topic').value
        rate = self.get_parameter('publish_rate').value
        
        # Publishers
        self.vision_pose_pub = self.create_publisher(
            PoseStamped,
            mavros_topic,
            10
        )
        
        self.odom_pub = self.create_publisher(
            Odometry,
            odom_topic,
            10
        )
        
        # Timer
        self.timer = self.create_timer(1.0 / rate, self.publish_dummy_pose)
        
        self.get_logger().info(f'Dummy Vision Pose Publisher started')
        self.get_logger().info(f'Publishing to: {mavros_topic}')
        self.get_logger().info(f'Publishing odometry to: {odom_topic}')
        self.get_logger().info(f'Rate: {rate} Hz')
    
    def publish_dummy_pose(self):
        """Publish dummy pose tại (0, 0, 0) với quaternion identity"""
        now = self.get_clock().now()
        
        # Vision Pose (PoseStamped) cho MAVROS
        vision_pose = PoseStamped()
        vision_pose.header.stamp = now.to_msg()
        vision_pose.header.frame_id = 'map'
        
        # Position: (0, 0, 0) - tại điểm xuất phát
        vision_pose.pose.position.x = 0.0
        vision_pose.pose.position.y = 0.0
        vision_pose.pose.position.z = 0.0
        
        # Orientation: Identity quaternion (no rotation)
        vision_pose.pose.orientation.x = 0.0
        vision_pose.pose.orientation.y = 0.0
        vision_pose.pose.orientation.z = 0.0
        vision_pose.pose.orientation.w = 1.0
        
        self.vision_pose_pub.publish(vision_pose)
        
        # Odometry (cho các node khác trong hệ thống)
        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = 'map'
        odom.child_frame_id = 'base_link'
        
        odom.pose.pose.position.x = 0.0
        odom.pose.pose.position.y = 0.0
        odom.pose.pose.position.z = 0.0
        
        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = 0.0
        odom.pose.pose.orientation.w = 1.0
        
        # Covariance (identity matrix * small value)
        odom.pose.covariance = [0.01] * 36  # 6x6 matrix
        
        self.odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = DummyVisionPose()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

