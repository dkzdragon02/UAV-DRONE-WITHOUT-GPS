#!/usr/bin/env python3
"""
ROS2 Evaluation Node
Monitor và evaluate performance của hệ thống
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Float64
from uav_vision.evaluation import SLAMEvaluator, NavigationEvaluator
import numpy as np
import time


class EvaluationNode(Node):
    """Evaluation node"""
    
    def __init__(self):
        super().__init__('evaluation_node')
        
        # Evaluators
        self.slam_evaluator = SLAMEvaluator()
        self.nav_evaluator = NavigationEvaluator()
        
        # Parameters
        self.declare_parameter('odom_topic', '/uav/slam/odometry')
        self.declare_parameter('pose_topic', '/uav/slam/pose')
        self.declare_parameter('report_topic', '/uav/evaluation/report')
        self.declare_parameter('metrics_topic', '/uav/evaluation/metrics')
        self.declare_parameter('report_interval', 10.0)  # seconds
        
        odom_topic = self.get_parameter('odom_topic').value
        pose_topic = self.get_parameter('pose_topic').value
        report_topic = self.get_parameter('report_topic').value
        metrics_topic = self.get_parameter('metrics_topic').value
        report_interval = self.get_parameter('report_interval').value
        
        # Subscribers
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
        
        # Publishers
        self.report_pub = self.create_publisher(String, report_topic, 10)
        self.metrics_pub = self.create_publisher(Float64, metrics_topic, 10)
        
        # Timer for reports
        self.report_timer = self.create_timer(report_interval, self.publish_report)
        
        self.last_report_time = time.time()
        
        self.get_logger().info('Evaluation Node started')
    
    def odom_callback(self, msg):
        """Process odometry"""
        pose = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        yaw = 2 * np.arctan2(q.z, q.w)
        pose = np.array([pose[0], pose[1], yaw])
        
        self.slam_evaluator.add_pose(pose)
    
    def pose_callback(self, msg):
        """Process pose"""
        pose = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            0.0  # yaw from quaternion if needed
        ])
        
        # Could add ground truth comparison here if available
        self.slam_evaluator.add_pose(pose)
    
    def publish_report(self):
        """Publish evaluation report"""
        report = self.slam_evaluator.get_report()
        
        msg = String()
        msg.data = report
        self.report_pub.publish(msg)
        
        # Also log
        self.get_logger().info('\n' + report)
        
        # Publish metrics
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

