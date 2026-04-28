import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import time

class DummyVisionPose(Node):
    def __init__(self):
        super().__init__('dummy_vision_pose')
        
        self.declare_parameter('mavros_vision_topic', '/mavros/vision_pose/pose')
        self.declare_parameter('vision_odom_input_topic', '/uav/vision/odometry')
        self.declare_parameter('vision_odom_output_topic', '/uav/vision/odometry_fused')
        self.declare_parameter('publish_rate', 30.0)  # Hz
        self.declare_parameter('stale_timeout', 2.0)  # seconds 
        
        mavros_topic = self.get_parameter('mavros_vision_topic').value
        odom_input_topic = self.get_parameter('vision_odom_input_topic').value
        odom_output_topic = self.get_parameter('vision_odom_output_topic').value
        rate = self.get_parameter('publish_rate').value
        self.stale_timeout = self.get_parameter('stale_timeout').value
        self.latest_odom = None
        self.last_odom_time = None
        self.using_real_odom = False
        self.stale_warned = False
        
        self.vision_pose_pub = self.create_publisher(
            PoseStamped,
            mavros_topic,
            10
        )
        
        self.odom_pub = self.create_publisher(
            Odometry,
            odom_output_topic,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            odom_input_topic,
            self.odom_callback,
            10
        )

        self.timer = self.create_timer(1.0 / rate, self.publish_pose)
        self.get_logger().info(f'Vision Pose Publisher started (upgraded)')
        self.get_logger().info(f'Publishing vision pose to: {mavros_topic}')
        self.get_logger().info(f'Listening for real VO on: {odom_input_topic}')
        self.get_logger().info(f'Publishing fused odom to: {odom_output_topic}')
        self.get_logger().info(f'Rate: {rate} Hz, stale timeout: {self.stale_timeout}s')
        self.get_logger().info(f'No real VO yet — using dummy pose (0,0,0) for EKF')
    
    def odom_callback(self, msg: Odometry):
        self.latest_odom = msg
        self.last_odom_time = time.time()
        
        if not self.using_real_odom:
            self.using_real_odom = True
            self.stale_warned = False
            self.get_logger().info('Receiving real vision odometry — forwarding to MAVROS')
    
    def _is_odom_stale(self) -> bool:
        if self.last_odom_time is None:
            return True
        return (time.time() - self.last_odom_time) > self.stale_timeout
    
    def publish_pose(self):
        now = self.get_clock().now()
        
        if self.latest_odom is not None and not self._is_odom_stale():
            self._publish_real_pose(now)
        else:
            if self.using_real_odom and not self.stale_warned:
                self.get_logger().warn(
                    f'Vision odometry stale (>{self.stale_timeout}s) — '
                    f'falling back to dummy pose (0,0,0). EKF will drift!'
                )
                self.stale_warned = True
                self.using_real_odom = False
            
            self._publish_dummy_pose(now)
    
    def _publish_real_pose(self, now):
        odom = self.latest_odom
        
        vision_pose = PoseStamped()
        vision_pose.header.stamp = now.to_msg()
        vision_pose.header.frame_id = 'map'
        vision_pose.pose.position.x = odom.pose.pose.position.x
        vision_pose.pose.position.y = odom.pose.pose.position.y
        vision_pose.pose.position.z = odom.pose.pose.position.z
        vision_pose.pose.orientation.x = odom.pose.pose.orientation.x
        vision_pose.pose.orientation.y = odom.pose.pose.orientation.y
        vision_pose.pose.orientation.z = odom.pose.pose.orientation.z
        vision_pose.pose.orientation.w = odom.pose.pose.orientation.w
        
        self.vision_pose_pub.publish(vision_pose)
    
    def _publish_dummy_pose(self, now):
        vision_pose = PoseStamped()
        vision_pose.header.stamp = now.to_msg()
        vision_pose.header.frame_id = 'map'
        vision_pose.pose.position.x = 0.0
        vision_pose.pose.position.y = 0.0
        vision_pose.pose.position.z = 0.0
        vision_pose.pose.orientation.x = 0.0
        vision_pose.pose.orientation.y = 0.0
        vision_pose.pose.orientation.z = 0.0
        vision_pose.pose.orientation.w = 1.0
        
        self.vision_pose_pub.publish(vision_pose)

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
        odom.pose.covariance = [0.1] * 36  
        
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
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
