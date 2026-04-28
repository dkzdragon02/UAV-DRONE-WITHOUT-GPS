import math
from typing import List
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import String, Bool, Float64
from .obstacle_avoidance import ObstacleAvoidance

class ObstacleAvoidanceNode(Node):
    def __init__(self) -> None:
        super().__init__("obstacle_avoidance_node")
        self.declare_parameter("odom_topic", "/uav/vision/odometry")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("depth_image_topic", "/camera/depth/image_raw")
        self.declare_parameter("goal_x", 5.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("max_velocity", 1.0)
        self.declare_parameter("output_topic", "/uav/obstacle_avoidance/velocity")
        self.declare_parameter("output_stamped", True)
        self.declare_parameter("correction_topic", "/uav/obstacle_avoidance/velocity_correction")
        self.declare_parameter("emergency_stop_topic", "/uav/obstacle_avoidance/emergency_stop")
        self.declare_parameter("nearest_dist_topic", "/uav/obstacle_avoidance/nearest_obstacle_dist")
        self.declare_parameter("safety_distance", 0.5)
        self.declare_parameter("emergency_distance", 0.3)
        self.declare_parameter("max_repulsive_distance", 2.0)
        self.declare_parameter("repulsive_gain", 1.5)
        self.declare_parameter("depth_obstacle_max_range", 3.0)
        self.declare_parameter("depth_num_columns", 8)  # sample N columns from depth image

        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        depth_image_topic = self.get_parameter("depth_image_topic").get_parameter_value().string_value
        self.max_velocity = float(self.get_parameter("max_velocity").value)
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.output_stamped = bool(self.get_parameter("output_stamped").value)
        correction_topic = self.get_parameter("correction_topic").get_parameter_value().string_value
        emergency_stop_topic = self.get_parameter("emergency_stop_topic").get_parameter_value().string_value
        nearest_dist_topic = self.get_parameter("nearest_dist_topic").get_parameter_value().string_value
        self.depth_max_range = float(self.get_parameter("depth_obstacle_max_range").value)
        self.depth_num_columns = int(self.get_parameter("depth_num_columns").value)

        self.goal = np.array(
            [
                float(self.get_parameter("goal_x").value),
                float(self.get_parameter("goal_y").value),
            ],
            dtype=float,
        )

        self.current_pos: np.ndarray | None = None
        self.current_altitude: float = 0.0
        self.obstacles: List[np.ndarray] = []
        self.flight_state: str = "idle"
        self.enabled = True  
        self._active_states = {"navigation", "manual_control", "hover", "rtl"}
        
        self.oa = ObstacleAvoidance(
            repulsive_gain=float(self.get_parameter("repulsive_gain").value),
        )
        self.oa.safety_distance = float(self.get_parameter("safety_distance").value)
        self.oa.emergency_distance = float(self.get_parameter("emergency_distance").value)
        self.oa.max_repulsive_distance = float(self.get_parameter("max_repulsive_distance").value)

        if self.output_stamped:
            self.cmd_vel_pub = self.create_publisher(TwistStamped, output_topic, 10)
        else:
            self.cmd_vel_pub = self.create_publisher(Twist, output_topic, 10)
        self.legacy_cmd_vel_pub = self.create_publisher(Twist, "/px4_mavlink/setpoint_vel", 10)
        self.correction_pub = self.create_publisher(TwistStamped, correction_topic, 10)
        self.emergency_stop_pub = self.create_publisher(Bool, emergency_stop_topic, 10)
        self.nearest_dist_pub = self.create_publisher(Float64, nearest_dist_topic, 10)
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, depth_image_topic, self.depth_callback, 10
        )
        self.state_sub = self.create_subscription(
            String, "/uav/state_machine/state", self.state_callback, 10
        )

        self.enable_sub = self.create_subscription(
            Bool, "/uav/obstacle_avoidance/enable", self.enable_callback, 10
        )

        self.timer = self.create_timer(0.05, self.timer_callback)  # 20 Hz
        self.get_logger().info(
            f"ObstacleAvoidanceNode started. "
            f"odom={odom_topic}, scan={scan_topic}, depth={depth_image_topic}, "
            f"max_velocity={self.max_velocity}, "
            f"safety_dist={self.oa.safety_distance}, emergency_dist={self.oa.emergency_distance}"
        )

    def odom_callback(self, msg: Odometry) -> None:
        self.current_pos = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y], dtype=float
        )
        self.current_altitude = float(msg.pose.pose.position.z)

    def scan_callback(self, msg: LaserScan) -> None:
        obstacles: List[np.ndarray] = []
        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max and r > 0.05:
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                if math.hypot(x, y) <= self.oa.max_repulsive_distance:
                    obstacles.append(np.array([x, y], dtype=float))
            angle += msg.angle_increment
        self.obstacles = obstacles
        self.oa.update_obstacle_history(obstacles)

    def depth_callback(self, msg: Image) -> None:
        if self.obstacles:
            return

        try:
            if msg.encoding == '32FC1':
                import struct
                step = msg.step
                width = msg.width
                height = msg.height
                mid_row = height // 2
                fov_h = 1.047   # horizontal FOV (radians)
                fov_v = 0.785   # vertical FOV (radians, ~45°)

                obstacles: List[np.ndarray] = []
                cols_to_sample = np.linspace(0, width - 1, self.depth_num_columns, dtype=int)
                rows_to_sample = [height // 4, mid_row, 3 * height // 4]

                for row in rows_to_sample:
                    for col in cols_to_sample:
                        offset = row * step + col * 4
                        if offset + 4 <= len(msg.data):
                            depth = struct.unpack('f', msg.data[offset:offset + 4])[0]
                            if 0.1 < depth < self.depth_max_range and math.isfinite(depth):
                                angle_h = (col / width - 0.5) * fov_h
                                angle_v = (row / height - 0.5) * fov_v
                                x = depth * math.cos(angle_h) * math.cos(angle_v)
                                y = depth * math.sin(angle_h)
                                z = self.current_altitude - depth * math.sin(angle_v)
                                obstacles.append(np.array([x, y, z], dtype=float))

                if obstacles:
                    self.obstacles = obstacles
                    self.oa.update_obstacle_history(obstacles)
        except Exception:
            pass  

    def state_callback(self, msg: String) -> None:
        self.flight_state = msg.data.lower().strip()

    def enable_callback(self, msg: Bool) -> None:
        self.enabled = msg.data
        self.get_logger().info(f"Obstacle avoidance {'ENABLED' if self.enabled else 'DISABLED'}")

    def timer_callback(self) -> None:
        if self.current_pos is None:
            return

        nearest_dist = self.oa.get_nearest_obstacle_distance(self.current_pos, self.obstacles)
        emergency = self.oa.is_emergency_stop(self.current_pos, self.obstacles)
        dist_msg = Float64()
        dist_msg.data = float(nearest_dist) if math.isfinite(nearest_dist) else 999.0
        self.nearest_dist_pub.publish(dist_msg)
        estop_msg = Bool()
        estop_msg.data = emergency and self.enabled
        self.emergency_stop_pub.publish(estop_msg)

        if not self.enabled or self.flight_state not in self._active_states:
            self._publish_zero_correction()
            return

        if np.linalg.norm(self.goal - self.current_pos) < 0.2:
            self._publish_zero_correction()
            return

        pos_3d = np.array([self.current_pos[0], self.current_pos[1], self.current_altitude])
        correction_3d = self.oa.compute_velocity_correction(
            pos_3d, self.obstacles, max_correction=self.max_velocity
        )
        corr_msg = TwistStamped()
        corr_msg.header.stamp = self.get_clock().now().to_msg()
        corr_msg.header.frame_id = "base_link"
        corr_msg.twist.linear.x = float(correction_3d[0])
        corr_msg.twist.linear.y = float(correction_3d[1])
        corr_msg.twist.linear.z = float(correction_3d[2]) if len(correction_3d) > 2 else 0.0
        self.correction_pub.publish(corr_msg)

        vel_xy = self.oa.avoid_obstacles(
            self.current_pos, self.goal, self.obstacles, max_velocity=self.max_velocity
        )

        if self.output_stamped:
            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.header.frame_id = "base_link"
            cmd.twist.linear.x = float(vel_xy[0])
            cmd.twist.linear.y = float(vel_xy[1])
            cmd.twist.linear.z = 0.0
            self.cmd_vel_pub.publish(cmd)
        else:
            cmd = Twist()
            cmd.linear.x = float(vel_xy[0])
            cmd.linear.y = float(vel_xy[1])
            cmd.linear.z = 0.0
            self.cmd_vel_pub.publish(cmd)

        legacy_cmd = Twist()
        legacy_cmd.linear.x = float(vel_xy[0])
        legacy_cmd.linear.y = float(vel_xy[1])
        legacy_cmd.linear.z = 0.0
        self.legacy_cmd_vel_pub.publish(legacy_cmd)

    def _publish_zero_correction(self) -> None:
        corr_msg = TwistStamped()
        corr_msg.header.stamp = self.get_clock().now().to_msg()
        corr_msg.header.frame_id = "base_link"
        corr_msg.twist.linear.x = 0.0
        corr_msg.twist.linear.y = 0.0
        corr_msg.twist.linear.z = 0.0
        self.correction_pub.publish(corr_msg)

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
