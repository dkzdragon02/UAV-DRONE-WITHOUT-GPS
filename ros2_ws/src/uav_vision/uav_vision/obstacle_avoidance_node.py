#!/usr/bin/env python3
"""
ROS2 node: Obstacle avoidance (2D potential field) -> velocity setpoint for PX4

Đọc:
- /uav/vision/odometry (nav_msgs/Odometry)
- /scan (sensor_msgs/LaserScan) – lidar 2D 360°

Xuất:
- /px4_mavlink/setpoint_vel (geometry_msgs/Twist) – vận tốc body-frame (x, y, z)

Node này chỉ tính toán và publish setpoint. PX4 thực thi qua px4_mavlink_bridge.
"""

import math
from typing import List

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from .obstacle_avoidance import ObstacleAvoidance


class ObstacleAvoidanceNode(Node):
    def __init__(self) -> None:
        super().__init__("obstacle_avoidance_node")

        # Parameters
        self.declare_parameter("odom_topic", "/uav/vision/odometry")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("goal_x", 5.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("max_velocity", 1.0)

        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.max_velocity = float(self.get_parameter("max_velocity").value)

        self.goal = np.array(
            [
                float(self.get_parameter("goal_x").value),
                float(self.get_parameter("goal_y").value),
            ],
            dtype=float,
        )

        # State
        self.current_pos: np.ndarray | None = None
        self.obstacles: List[np.ndarray] = []

        # Core algorithm
        self.oa = ObstacleAvoidance()

        # ROS interfaces
        self.cmd_vel_pub = self.create_publisher(
            Twist, "/px4_mavlink/setpoint_vel", 10
        )

        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, 10
        )

        # Main timer (20 Hz)
        self.timer = self.create_timer(0.05, self.timer_callback)

        self.get_logger().info(
            f"ObstacleAvoidanceNode started. "
            f"odom_topic={odom_topic}, scan_topic={scan_topic}, "
            f"goal=({self.goal[0]}, {self.goal[1]}), max_velocity={self.max_velocity}"
        )

    def odom_callback(self, msg: Odometry) -> None:
        self.current_pos = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y], dtype=float
        )

    def scan_callback(self, msg: LaserScan) -> None:
        obstacles: List[np.ndarray] = []
        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max and r > 0.05:
                # Convert polar (r, angle) -> Cartesian (x, y) in lidar frame
                x = r * math.cos(angle)
                y = r * math.sin(angle)

                # Bỏ xa hơn max_repulsive_distance để giảm noise
                if math.hypot(x, y) <= self.oa.max_repulsive_distance:
                    obstacles.append(np.array([x, y], dtype=float))

            angle += msg.angle_increment

        self.obstacles = obstacles

    def timer_callback(self) -> None:
        if self.current_pos is None:
            return

        # Nếu đã gần goal thì không cần di chuyển
        if np.linalg.norm(self.goal - self.current_pos) < 0.2:
            return

        # Tính vận tốc tránh vật cản (2D: x, y)
        vel_xy = self.oa.avoid_obstacles(
            self.current_pos, self.goal, self.obstacles, max_velocity=self.max_velocity
        )

        cmd = Twist()
        cmd.linear.x = float(vel_xy[0])
        cmd.linear.y = float(vel_xy[1])
        cmd.linear.z = 0.0  # Giữ độ cao, có thể mở rộng sau

        self.cmd_vel_pub.publish(cmd)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObstacleAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


