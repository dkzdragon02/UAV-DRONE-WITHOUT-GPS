import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from uav_vision.qos_profiles import mavros_qos as make_mavros_qos
from geometry_msgs.msg import PoseStamped, TwistStamped, Point, Twist
from std_msgs.msg import String, Bool, Float64
import numpy as np
import math
import time

class PositionPID:
    def __init__(self, kp=1.0, ki=0.05, kd=0.3, max_integral=2.0, max_output=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        self.max_output = max_output
        self.integral = np.array([0.0, 0.0, 0.0])
        self.prev_error = np.array([0.0, 0.0, 0.0])
        self.last_time = None

    def reset(self):
        self.integral = np.array([0.0, 0.0, 0.0])
        self.prev_error = np.array([0.0, 0.0, 0.0])
        self.last_time = None

    def compute(self, error: np.ndarray) -> np.ndarray:
        current_time = time.time()

        if self.last_time is None:
            self.last_time = current_time
            self.prev_error = error.copy()
            output = self.kp * error
            return self._clamp(output)

        dt = current_time - self.last_time
        if dt <= 0 or dt > 1.0:
            self.last_time = current_time
            self.prev_error = error.copy()
            return self._clamp(self.kp * error)

        self.last_time = current_time

        p_term = self.kp * error

        self.integral += error * dt
        for i in range(3):
            self.integral[i] = np.clip(
                self.integral[i], -self.max_integral, self.max_integral
            )
        i_term = self.ki * self.integral
        d_term = self.kd * (error - self.prev_error) / dt
        self.prev_error = error.copy()

        output = p_term + i_term + d_term
        return self._clamp(output)

    def _clamp(self, output: np.ndarray) -> np.ndarray:
        speed = np.linalg.norm(output)
        if speed > self.max_output:
            output = output / speed * self.max_output
        return output


class PX4Controller(Node):
    def __init__(self):
        super().__init__('px4_controller')
        self.declare_parameter('path_topic', '/uav/path_planner/path')
        self.declare_parameter('pose_topic', '/mavros/local_position/pose')
        self.declare_parameter('odom_topic', '/uav/vision/odometry')
        self.declare_parameter('state_topic', '/uav/state_machine/state')
        self.declare_parameter('velocity_topic', '/uav/px4_controller/velocity')
        self.declare_parameter('position_target_topic', '/uav/px4_controller/position_target')
        self.declare_parameter('altitude_topic', '/uav/px4_controller/altitude')
        self.declare_parameter('max_velocity', 1.0)         # m/s
        self.declare_parameter('takeoff_velocity', 0.8)     # m/s (vertical)
        self.declare_parameter('landing_velocity', 0.3)     # m/s (vertical, positive = descend speed)
        self.declare_parameter('target_altitude', 2.0)      # meters
        self.declare_parameter('waypoint_tolerance', 0.5)   # meters
        self.declare_parameter('lookahead_distance', 1.0)   # meters
        self.declare_parameter('manual_velocity_topic', '/uav/manual_velocity')
        self.declare_parameter('manual_max_speed', 1.0)     # m/s clamp for manual
        self.declare_parameter('landing_threshold', 0.15)   # meters — considered landed
        self.declare_parameter('obstacle_avoidance_gain', 1.0)  # 0.0 = off, 1.0 = full
        self.declare_parameter('obstacle_correction_topic', '/uav/obstacle_avoidance/velocity_correction')
        self.declare_parameter('obstacle_estop_topic', '/uav/obstacle_avoidance/emergency_stop')
        self.declare_parameter('obstacle_nearest_dist_topic', '/uav/obstacle_avoidance/nearest_obstacle_dist')

        # ── Position PID parameters ──
        self.declare_parameter('hover_kp', 1.0)
        self.declare_parameter('hover_ki', 0.05)
        self.declare_parameter('hover_kd', 0.3)
        self.declare_parameter('hover_max_integral', 2.0)
        self.declare_parameter('hover_max_correction', 0.5)  # m/s — max correction speed

        path_topic = self.get_parameter('path_topic').value
        pose_topic = self.get_parameter('pose_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        state_topic = self.get_parameter('state_topic').value

        self.max_velocity = self.get_parameter('max_velocity').value
        self.takeoff_velocity = self.get_parameter('takeoff_velocity').value
        self.landing_velocity = self.get_parameter('landing_velocity').value
        self.target_altitude = self.get_parameter('target_altitude').value
        self.waypoint_tolerance = self.get_parameter('waypoint_tolerance').value
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.landing_threshold = self.get_parameter('landing_threshold').value
        self.manual_max_speed = self.get_parameter('manual_max_speed').value
        self.obstacle_avoidance_gain = self.get_parameter('obstacle_avoidance_gain').value
        self.current_pose = None
        self.current_altitude = 0.0
        self.current_path = None
        self.current_state = "idle"
        self.current_path_index = 0
        self.manual_velocity = np.array([0.0, 0.0, 0.0])
        self.manual_yaw_rate = 0.0
        self.avoidance_correction = np.array([0.0, 0.0, 0.0])
        self.obstacle_emergency_stop = False
        self.nearest_obstacle_dist = float('inf')

        hover_kp = self.get_parameter('hover_kp').value
        hover_ki = self.get_parameter('hover_ki').value
        hover_kd = self.get_parameter('hover_kd').value
        hover_max_integral = self.get_parameter('hover_max_integral').value
        hover_max_correction = self.get_parameter('hover_max_correction').value

        self.hover_pid = PositionPID(
            kp=hover_kp,
            ki=hover_ki,
            kd=hover_kd,
            max_integral=hover_max_integral,
            max_output=hover_max_correction,
        )
        self.hover_target = None  # Latched position when entering hover

        self.path_sub = self.create_subscription(
            Path, path_topic, self.path_callback, 10
        )

        mavros_qos = make_mavros_qos()

        self.pose_sub = self.create_subscription(
            PoseStamped, pose_topic, self.pose_callback, mavros_qos
        )

        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 10
        )

        self.state_sub = self.create_subscription(
            String, state_topic, self.state_callback, 10
        )

        manual_vel_topic = self.get_parameter('manual_velocity_topic').value
        self.manual_vel_sub = self.create_subscription(
            TwistStamped, manual_vel_topic, self.manual_vel_callback, 10
        )

        obs_corr_topic = self.get_parameter('obstacle_correction_topic').value
        obs_estop_topic = self.get_parameter('obstacle_estop_topic').value
        obs_dist_topic = self.get_parameter('obstacle_nearest_dist_topic').value

        self.obs_correction_sub = self.create_subscription(
            TwistStamped, obs_corr_topic, self.obs_correction_callback, 10
        )
        self.obs_estop_sub = self.create_subscription(
            Bool, obs_estop_topic, self.obs_estop_callback, 10
        )
        self.obs_dist_sub = self.create_subscription(
            Float64, obs_dist_topic, self.obs_dist_callback, 10
        )

        self.velocity_pub = self.create_publisher(
            TwistStamped,
            self.get_parameter('velocity_topic').value,
            10
        )

        self.position_target_pub = self.create_publisher(
            PoseStamped,
            self.get_parameter('position_target_topic').value,
            10
        )

        self.altitude_pub = self.create_publisher(
            Float64,
            self.get_parameter('altitude_topic').value,
            10
        )

        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.alt_timer = self.create_timer(0.2, self.publish_altitude)
        self.get_logger().info('PX4 Controller started')
        self.get_logger().info(f'  Target altitude: {self.target_altitude}m')
        self.get_logger().info(f'  Takeoff velocity: {self.takeoff_velocity} m/s')
        self.get_logger().info(f'  Max velocity: {self.max_velocity} m/s')
        self.get_logger().info(f'  Manual velocity topic: {manual_vel_topic}')
        self.get_logger().info(f'  Manual max speed: {self.manual_max_speed} m/s')
        self.get_logger().info(f'  Obstacle avoidance gain: {self.obstacle_avoidance_gain}')
        self.get_logger().info(
            f'  Hover PID: kp={hover_kp}, ki={hover_ki}, kd={hover_kd}, '
            f'max_correction={hover_max_correction}m/s'
        )

    def path_callback(self, msg):
        self.current_path = msg
        self.current_path_index = 0
        self.get_logger().info(f'Received new path with {len(msg.poses)} waypoints')

    def pose_callback(self, msg):
        self.current_pose = msg
        self.current_altitude = msg.pose.position.z

    def odom_callback(self, msg):
        if self.current_pose is None:

            pose = PoseStamped()
            pose.header = msg.header
            pose.pose = msg.pose.pose
            self.current_pose = pose
            self.current_altitude = msg.pose.pose.position.z

    def state_callback(self, msg):
        new_state = msg.data.lower()
        if new_state != self.current_state:
            self.get_logger().info(f'State changed: {self.current_state} → {new_state}')

            if self.current_state == 'hover' and new_state != 'hover':
                self.hover_target = None
                self.hover_pid.reset()

            if self.current_state == 'manual_control' and new_state != 'manual_control':
                self.manual_velocity = np.array([0.0, 0.0, 0.0])
                self.manual_yaw_rate = 0.0

            if new_state == 'hover' and self.current_state != 'hover':
                self.hover_target = self._get_current_position()
                self.hover_pid.reset()
                if self.hover_target is not None:
                    self.get_logger().info(
                        f'Hover PID: target latched at '
                        f'[{self.hover_target[0]:.2f}, {self.hover_target[1]:.2f}, '
                        f'{self.hover_target[2]:.2f}]'
                    )

            self.current_state = new_state

    def manual_vel_callback(self, msg):
        self.manual_velocity = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z,
        ])
        self.manual_yaw_rate = msg.twist.angular.z

    def obs_correction_callback(self, msg):
        self.avoidance_correction = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z,
        ])

    def obs_estop_callback(self, msg):
        if msg.data and not self.obstacle_emergency_stop:
            self.get_logger().warn('OBSTACLE EMERGENCY STOP — obstacle too close!')
        self.obstacle_emergency_stop = msg.data

    def obs_dist_callback(self, msg):
        self.nearest_obstacle_dist = msg.data

    def _get_current_position(self) -> np.ndarray:
        if self.current_pose is None:
            return None
        p = self.current_pose.pose.position
        return np.array([p.x, p.y, p.z])

    def control_loop(self):
        if self.current_pose is None:
            return

        if self.obstacle_emergency_stop and self.current_state in (
            'navigation', 'manual_control', 'hover'
        ):
            self.get_logger().warn(
                f'Obstacle E-STOP active (nearest: {self.nearest_obstacle_dist:.2f}m) — holding position',
                throttle_duration_sec=2.0
            )
            self.publish_velocity(np.array([0.0, 0.0, 0.0]))
            return

        if self.current_state == "navigation" and self.current_path:
            self.follow_path()
        elif self.current_state == "hover":
            self.hover_control()
        elif self.current_state == "takeoff":
            self.takeoff_control()
        elif self.current_state == "landing":
            self.landing_control()
        elif self.current_state == "rtl":
            self.rtl_control()
        elif self.current_state == "manual_control":
            self.manual_control()
        elif self.current_state == "emergency":
            self.emergency_control()

    def takeoff_control(self):
        altitude_error = self.target_altitude - self.current_altitude

        if altitude_error <= 0.1:
            self.get_logger().info(
                f'Target altitude reached: {self.current_altitude:.2f}m '
                f'(target: {self.target_altitude:.2f}m)'
            )
            self.publish_velocity(np.array([0.0, 0.0, 0.0]))
            return

        vz = min(self.takeoff_velocity, max(0.2, altitude_error * 0.5))

        velocity = np.array([0.0, 0.0, vz])
        self.publish_velocity(velocity)

    def hover_control(self):
        current_pos = self._get_current_position()

        if current_pos is None:
            self.publish_velocity(np.array([0.0, 0.0, 0.0]))
            return

        if self.hover_target is None:
            self.hover_target = current_pos.copy()
            self.hover_pid.reset()
            self.get_logger().info(
                f'Hover PID: target latched at '
                f'[{self.hover_target[0]:.2f}, {self.hover_target[1]:.2f}, '
                f'{self.hover_target[2]:.2f}]'
            )

        error = self.hover_target - current_pos
        velocity = self.hover_pid.compute(error)
        self.publish_velocity(velocity)

    def landing_control(self):
        if self.current_altitude <= self.landing_threshold:
            self.publish_velocity(np.array([0.0, 0.0, 0.0]))
            return
        vz = -min(self.landing_velocity, max(0.1, self.current_altitude * 0.3))

        velocity = np.array([0.0, 0.0, vz])
        self.publish_velocity(velocity)

    def rtl_control(self):
        target = PoseStamped()
        target.pose.position.x = 0.0
        target.pose.position.y = 0.0
        target.pose.position.z = self.target_altitude
        velocity = self.calculate_velocity(target)
        velocity = self._apply_avoidance(velocity)
        self.publish_velocity(velocity)

    def emergency_control(self):
        if self.current_altitude > self.landing_threshold:
            self.publish_velocity(np.array([0.0, 0.0, -0.5]))
        else:
            self.publish_velocity(np.array([0.0, 0.0, 0.0]))

    def manual_control(self):
        velocity = self.manual_velocity.copy()
        speed = np.linalg.norm(velocity)
        if speed > self.manual_max_speed:
            velocity = velocity / speed * self.manual_max_speed
        velocity = self._apply_avoidance(velocity)
        yaw_rate = np.clip(self.manual_yaw_rate, -1.0, 1.0)
        self.publish_velocity(velocity, yaw_rate=yaw_rate)

    def follow_path(self):
        if not self.current_path or len(self.current_path.poses) == 0:
            return

        closest_index = self.find_closest_path_point()
        lookahead_index = self.find_lookahead_point(closest_index)
        if lookahead_index >= len(self.current_path.poses):
            lookahead_index = len(self.current_path.poses) - 1
        target_pose = self.current_path.poses[lookahead_index]
        last_pose = self.current_path.poses[-1]
        dist_to_end = self.distance(
            self.current_pose.pose.position, last_pose.pose.position
        )
        if dist_to_end < self.waypoint_tolerance:
            self.get_logger().info('Path complete — all waypoints reached')
            self.publish_velocity(np.array([0.0, 0.0, 0.0]))
            return
        velocity = self.calculate_velocity(target_pose)
        velocity = self._apply_avoidance(velocity)
        self.publish_velocity(velocity)

    def find_closest_path_point(self) -> int:
        if not self.current_path or len(self.current_path.poses) == 0:
            return 0

        min_dist = float('inf')
        closest_index = 0
        current_pos = self.current_pose.pose.position
        for i, path_pose in enumerate(self.current_path.poses):
            dist = self.distance(current_pos, path_pose.pose.position)
            if dist < min_dist:
                min_dist = dist
                closest_index = i
        return closest_index

    def find_lookahead_point(self, start_index: int) -> int:
        if not self.current_path or len(self.current_path.poses) == 0:
            return 0
        current_pos = self.current_pose.pose.position
        for i in range(start_index, len(self.current_path.poses)):
            path_pose = self.current_path.poses[i]
            dist = self.distance(current_pos, path_pose.pose.position)

            if dist >= self.lookahead_distance:
                return i
        return len(self.current_path.poses) - 1

    def calculate_velocity(self, target_pose: PoseStamped) -> np.ndarray:
        if self.current_pose is None:
            return np.array([0.0, 0.0, 0.0])
        current_pos = self.current_pose.pose.position
        target_pos = target_pose.pose.position
        dx = target_pos.x - current_pos.x
        dy = target_pos.y - current_pos.y
        dz = target_pos.z - current_pos.z
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        if distance < self.waypoint_tolerance:
            return np.array([0.0, 0.0, 0.0])
        velocity = np.array([dx, dy, dz]) / distance * self.max_velocity
        speed = np.linalg.norm(velocity)
        if speed > self.max_velocity:
            velocity = velocity / speed * self.max_velocity
        return velocity

    def _apply_avoidance(self, planned_velocity: np.ndarray) -> np.ndarray:
        if self.obstacle_avoidance_gain <= 0.0:
            return planned_velocity

        correction = self.avoidance_correction * self.obstacle_avoidance_gain

        blended = planned_velocity.copy()
        blended[0] += correction[0]
        blended[1] += correction[1]
        if len(correction) > 2 and abs(correction[2]) > 0.01:
            blended[2] += correction[2]

        speed = np.linalg.norm(blended)
        if speed > self.max_velocity:
            blended = blended / speed * self.max_velocity

        return blended

    def publish_velocity(self, velocity: np.ndarray, yaw_rate: float = 0.0):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = float(velocity[0])
        msg.twist.linear.y = float(velocity[1])
        msg.twist.linear.z = float(velocity[2])
        msg.twist.angular.z = float(yaw_rate)

        self.velocity_pub.publish(msg)

    def publish_altitude(self):
        msg = Float64()
        msg.data = float(self.current_altitude)
        self.altitude_pub.publish(msg)

    def distance(self, p1: Point, p2: Point) -> float:
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        dz = p2.z - p1.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)


def main(args=None):
    rclpy.init(args=args)
    node = PX4Controller()

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
