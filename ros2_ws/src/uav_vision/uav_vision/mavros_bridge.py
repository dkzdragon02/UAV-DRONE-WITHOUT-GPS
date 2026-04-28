import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from mavros_msgs.srv import SetMode, CommandBool
import numpy as np

class MAVROSBridge(Node):
    def __init__(self):
        super().__init__('mavros_bridge')
        
        self.declare_parameter('vision_odom_topic', '/uav/vision/odometry')
        self.declare_parameter('mavros_vision_topic', '/mavros/vision_pose/pose')
        self.declare_parameter('frame_id', 'vision_odom')
        
        vision_topic = self.get_parameter('vision_odom_topic').value
        mavros_topic = self.get_parameter('mavros_vision_topic').value
        
        self.vision_pose_pub = self.create_publisher(
            PoseStamped,
            mavros_topic,
            10
        )
        
        self.vision_odom_pub = self.create_publisher(
            Odometry,
            '/mavros/odometry/out',
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            vision_topic,
            self.odom_callback,
            10
        )
        
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.get_logger().info('MAVROS Bridge started')
        self.get_logger().info(f'Subscribing to: {vision_topic}')
        self.get_logger().info(f'Publishing to: {mavros_topic}')
    
    def odom_callback(self, msg):
        try:
            vision_pose = PoseStamped()
            vision_pose.header.stamp = msg.header.stamp
            vision_pose.header.frame_id = msg.header.frame_id
            vision_pose.pose.position.x = msg.pose.pose.position.x
            vision_pose.pose.position.y = msg.pose.pose.position.y
            vision_pose.pose.position.z = msg.pose.pose.position.z
            vision_pose.pose.orientation = msg.pose.pose.orientation
            
            self.vision_pose_pub.publish(vision_pose)
            self.vision_odom_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in odom callback: {e}')
    
    def set_mode(self, mode: str):
        if not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('MAVROS set_mode service not available')
            return False
        
        request = SetMode.Request()
        request.custom_mode = mode
        
        future = self.set_mode_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result().mode_sent:
            self.get_logger().info(f'Flight mode set to: {mode}')
            return True
        else:
            self.get_logger().warn(f'Failed to set flight mode: {mode}')
            return False
    
    def arm(self, arm: bool):
        if not self.arming_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('MAVROS arming service not available')
            return False
        
        request = CommandBool.Request()
        request.value = arm
        
        future = self.arming_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result().success:
            self.get_logger().info(f'Vehicle {"armed" if arm else "disarmed"}')
            return True
        else:
            self.get_logger().warn(f'Failed to {"arm" if arm else "disarm"} vehicle')
            return False

def main(args=None):
    rclpy.init(args=args)
    node = MAVROSBridge()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

