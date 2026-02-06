#!/usr/bin/env python3
"""
Mock Lidar Publisher - Tạo dữ liệu LaserScan giả để test obstacle avoidance
Khi không có lidar thật trong Gazebo, node này sẽ publish scan data giả
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math


class MockLidarPublisher(Node):
    def __init__(self):
        super().__init__('mock_lidar_publisher')
        
        # Publisher
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        
        # Timer (10 Hz)
        self.timer = self.create_timer(0.1, self.publish_scan)
        
        self.get_logger().info('MockLidarPublisher started. Publishing to /scan')
    
    def publish_scan(self):
        """Publish mock LaserScan data"""
        scan_msg = LaserScan()
        
        # Header
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = 'base_link'  # hoặc 'laser_frame'
        
        # Scan parameters (360° lidar)
        scan_msg.angle_min = -math.pi
        scan_msg.angle_max = math.pi
        scan_msg.angle_increment = 2.0 * math.pi / 360.0  # 360 points
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 0.1  # 10 Hz
        
        # Range parameters
        scan_msg.range_min = 0.1
        scan_msg.range_max = 10.0
        
        # Generate ranges (mock: no obstacles, all max range)
        scan_msg.ranges = [10.0] * 360
        
        # Intensities (optional)
        scan_msg.intensities = [1.0] * 360
        
        self.scan_pub.publish(scan_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MockLidarPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

