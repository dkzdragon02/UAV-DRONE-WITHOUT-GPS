import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math

class MockLidarPublisher(Node):
    def __init__(self):
        super().__init__('mock_lidar_publisher')
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.timer = self.create_timer(0.1, self.publish_scan)
        self.get_logger().info('MockLidarPublisher started. Publishing to /scan')
    
    def publish_scan(self):
        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = 'base_link'              # hoặc 'laser_frame'
        scan_msg.angle_min = -math.pi
        scan_msg.angle_max = math.pi
        scan_msg.angle_increment = 2.0 * math.pi / 360.0    # 360 points
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 0.1                            # 10 Hz
        scan_msg.range_min = 0.1
        scan_msg.range_max = 10.0
        scan_msg.ranges = [10.0] * 360
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

