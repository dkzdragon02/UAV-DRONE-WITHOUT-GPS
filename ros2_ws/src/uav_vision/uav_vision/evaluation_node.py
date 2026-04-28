import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Float64
from uav_vision.evaluation import SLAMEvaluator, NavigationEvaluator
import numpy as np
import time

class EvaluationNode(Node):    
    def __init__(self):
        super().__init__('evaluation_node')
        self.slam_evaluator = SLAMEvaluator()
        self.nav_evaluator = NavigationEvaluator()
        self.declare_parameter('odom_topic', '/uav/slam/odometry')
        self.declare_parameter('pose_topic', '/uav/slam/pose')
        self.declare_parameter('report_topic', '/uav/evaluation/report')
        self.declare_parameter('metrics_topic', '/uav/evaluation/metrics')
        self.declare_parameter('report_interval', 10.0)                         # seconds
        
        odom_topic = self.get_parameter('odom_topic').value
        pose_topic = self.get_parameter('pose_topic').value
        report_topic = self.get_parameter('report_topic').value
        metrics_topic = self.get_parameter('metrics_topic').value
        report_interval = self.get_parameter('report_interval').value
        
        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            10
        )
        
        self.pose_sub = self.create_subscription(
            PoseStamped,
            pose_topic,
            self.pose_callback,
            10
        )
        
        self.report_pub = self.create_publisher(String, report_topic, 10)
        self.metrics_pub = self.create_publisher(Float64, metrics_topic, 10)
        self.report_timer = self.create_timer(report_interval, self.publish_report) # Timer for reports
        self.last_report_time = time.time()
        self.get_logger().info('Evaluation Node started')
    
    def odom_callback(self, msg):
        pose = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        
        q = msg.pose.pose.orientation
        yaw = 2 * np.arctan2(q.z, q.w)
        pose = np.array([pose[0], pose[1], yaw])
        
        self.slam_evaluator.add_pose(pose)
    
    def pose_callback(self, msg)
        pose = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            0.0 
        ])
        
        self.slam_evaluator.add_pose(pose)
    
    def publish_report(self):
        report = self.slam_evaluator.get_report()
        
        msg = String()
        msg.data = report
        self.report_pub.publish(msg)
        self.get_logger().info('\n' + report)
        
        metrics = self.slam_evaluator.compute_metrics()
        if 'mae_position' in metrics:
            metrics_msg = Float64()
            metrics_msg.data = metrics['mae_position']
            self.metrics_pub.publish(metrics_msg)


def main(args=None):
    rclpy.init(args=args)
    node = EvaluationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

