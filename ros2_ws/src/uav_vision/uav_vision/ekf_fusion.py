import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float64
import numpy as np
import math
import time
import threading

class EKFFusion(Node):
    def __init__(self):
        super().__init__('ekf_fusion')
        self.declare_parameter('vo_topic', '/uav/vision/odometry')
        self.declare_parameter('imu_topic', '/mavros/imu/data')
        self.declare_parameter('fused_odom_topic', '/uav/fused_odometry')
        self.declare_parameter('fused_pose_topic', '/uav/fused_pose')
        self.declare_parameter('publish_rate', 50.0)                 # Hz
        self.declare_parameter('accel_noise_std', 0.5)               # m/s²
        self.declare_parameter('gyro_noise_std', 0.01)               # rad/s
        self.declare_parameter('accel_bias_noise_std', 0.001)        # m/s²/√s
        self.declare_parameter('gyro_bias_noise_std', 0.0001)        # rad/s/√s
        self.declare_parameter('vo_position_noise_std', 5.0)         # m — high for synthetic/monocular
        self.declare_parameter('mahalanobis_threshold', 7.81)        # χ²(3, 0.95)
        self.declare_parameter('mahalanobis_hard_limit', 15.0)
        self.declare_parameter('soft_gate_enabled', True)
        self.declare_parameter('max_position_innovation', 50.0)      # m — absolute check (Mahalanobis does real gating)
        self.declare_parameter('innovation_gate_min', 3.0)           # m — tightest gate after convergence
        self.declare_parameter('innovation_gate_scale', 3.0)         # multiplier on position uncertainty
        self.declare_parameter('max_velocity_sanity', 3.0)           # m/s
        self.declare_parameter('max_attitude_deg', 60.0)             # ° — roll/pitch sanity
        self.declare_parameter('p_diagonal_max', 20.0)               # Covariance bound
        self.declare_parameter('innovation_window_size', 5)          # Consecutive rejection limit
        self.declare_parameter('max_frame_delta', 2.0)               # m — max per-frame VO displacement
        self.declare_parameter('vo_mode', 'incremental')             # 'incremental' or 'absolute'
        
        vo_topic = self.get_parameter('vo_topic').value
        imu_topic = self.get_parameter('imu_topic').value
        fused_odom_topic = self.get_parameter('fused_odom_topic').value
        fused_pose_topic = self.get_parameter('fused_pose_topic').value
        publish_rate = self.get_parameter('publish_rate').value

        self.accel_noise = self.get_parameter('accel_noise_std').value
        self.gyro_noise = self.get_parameter('gyro_noise_std').value
        self.accel_bias_noise = self.get_parameter('accel_bias_noise_std').value
        self.gyro_bias_noise = self.get_parameter('gyro_bias_noise_std').value
        self.vo_pos_noise = self.get_parameter('vo_position_noise_std').value
        self.mahal_threshold = self.get_parameter('mahalanobis_threshold').value
        self.mahal_hard_limit = self.get_parameter('mahalanobis_hard_limit').value
        self.soft_gate_enabled = self.get_parameter('soft_gate_enabled').value
        self.max_pos_innov = self.get_parameter('max_position_innovation').value
        self.innov_gate_min = self.get_parameter('innovation_gate_min').value
        self.innov_gate_scale = self.get_parameter('innovation_gate_scale').value
        self.max_vel_sanity = self.get_parameter('max_velocity_sanity').value
        self.max_attitude_rad = math.radians(self.get_parameter('max_attitude_deg').value)
        self.p_diag_max = self.get_parameter('p_diagonal_max').value
        self.innov_window_size = self.get_parameter('innovation_window_size').value
        self.max_frame_delta = self.get_parameter('max_frame_delta').value
        self.vo_mode = self.get_parameter('vo_mode').value
        self.state = np.zeros(16)
        self.state[6] = 1.0                     # qw = 1 (identity quaternion)
        self.P = np.eye(15) * 0.1
        self.P[0:3, 0:3] = np.eye(3) * 0.5      # Position
        self.P[3:6, 3:6] = np.eye(3) * 0.3      # Velocity
        self.P[6:9, 6:9] = np.eye(3) * 0.01     # Orientation — tight, IMU-driven
        self.P[9:12, 9:12] = np.eye(3) * 0.01   # Accel bias
        self.P[12:15, 12:15] = np.eye(3) * 0.001  # Gyro bias

        self._build_noise_matrices()
        self.gravity = np.array([0.0, 0.0, 9.81])   # NED: gravity points down (+Z)
        self.last_imu_stamp = None                  # ROS timestamp (nanoseconds)
        self.last_vo_time = None
        self.initialized = False
        self._vo_initialized = False                # True after first VO sets position
        self.lock = threading.Lock()
        self.imu_count = 0
        self.vo_count = 0
        self.rejected_vo_count = 0
        self.soft_gated_vo_count = 0
        self.abs_rejected_vo_count = 0
        self.vel_clamped_count = 0
        self.attitude_reset_count = 0
        self._consecutive_large_innov = 0
        self._recent_vo_accepted = 0
        self._recent_vo_rejected = 0
        self._adaptive_vo_scale = 1.0
        self._prev_vo_pos = None              # Previous VO position (for incremental mode)
        self._vo_accepted_count = 0           # VO updates that were applied
        
        self.vo_sub = self.create_subscription(
            Odometry, vo_topic, self.vo_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, imu_topic, self.imu_callback,
            qos_profile=qos_profile_sensor_data)

        self.fused_odom_pub = self.create_publisher(Odometry, fused_odom_topic, 10)
        self.fused_pose_pub = self.create_publisher(PoseStamped, fused_pose_topic, 10)
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_fused_state)
        self.status_timer = self.create_timer(5.0, self.log_status)
        self.get_logger().info('EKF Fusion Node started (position-only VO, IMU attitude)')
        self.get_logger().info(f'  VO topic: {vo_topic}')
        self.get_logger().info(f'  IMU topic: {imu_topic}')
        self.get_logger().info(f'  VO mode: {self.vo_mode} (frame delta → position update)')
        self.get_logger().info(f'  VO noise: {self.vo_pos_noise}m, max_frame_delta: {self.max_frame_delta}m')
        self.get_logger().info(f'  Innovation gate: adaptive min={self.innov_gate_min}m, '
                             f'scale={self.innov_gate_scale}x, hard={self.max_pos_innov}m')
        self.get_logger().info(f'  Max velocity: {self.max_vel_sanity}m/s')
        self.get_logger().info(f'  Max attitude: {math.degrees(self.max_attitude_rad):.0f}°')
        self.get_logger().info(f'  Mahalanobis: threshold={self.mahal_threshold}, hard={self.mahal_hard_limit}')

    def _build_noise_matrices(self):
        self.Q = np.zeros((15, 15))
        self.Q[0:3, 0:3] = np.eye(3) * (self.accel_noise ** 2) * 0.01    # Position
        self.Q[3:6, 3:6] = np.eye(3) * (self.accel_noise ** 2)           # Velocity
        self.Q[6:9, 6:9] = np.eye(3) * (self.gyro_noise ** 2)            # Orientation
        self.Q[9:12, 9:12] = np.eye(3) * (self.accel_bias_noise ** 2)    # Accel bias
        self.Q[12:15, 12:15] = np.eye(3) * (self.gyro_bias_noise ** 2)   # Gyro bias
        self.R_pos = np.eye(3) * (self.vo_pos_noise ** 2)

    @staticmethod
    def _quat_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    @staticmethod
    def _quat_normalize(q):
        n = np.linalg.norm(q)
        if n < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / n

    @staticmethod
    def _quat_conjugate(q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def _quat_to_rotation(q):
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ])

    @staticmethod
    def _rotation_vector_to_quat(rv):
        angle = np.linalg.norm(rv)
        if angle < 1e-10:
            return np.array([1.0, rv[0]*0.5, rv[1]*0.5, rv[2]*0.5])
        half = angle * 0.5
        s = math.sin(half) / angle
        return np.array([math.cos(half), rv[0]*s, rv[1]*s, rv[2]*s])

    @staticmethod
    def _quat_to_euler(q):
        w, x, y, z = q
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    @staticmethod
    def _euler_to_quaternion(roll, pitch, yaw):
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return np.array([qw, qx, qy, qz])

    @staticmethod
    def _ros_quat_to_internal(qx, qy, qz, qw):
        return np.array([qw, qx, qy, qz])

    @staticmethod
    def _skew(v):
        return np.array([
            [    0, -v[2],  v[1]],
            [ v[2],     0, -v[0]],
            [-v[1],  v[0],     0],
        ])

    @property
    def position(self):
        return self.state[0:3]

    @property
    def velocity(self):
        return self.state[3:6]

    @property
    def quaternion(self):
        """[w, x, y, z]"""
        return self.state[6:10]

    @property
    def accel_bias(self):
        return self.state[10:13]

    @property
    def gyro_bias(self):
        return self.state[13:16]

    def imu_callback(self, msg: Imu):
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if stamp_ns == 0:
            stamp_ns = int(time.time() * 1e9)

        with self.lock:
            q = msg.orientation
            if abs(q.w) > 0.01 or abs(q.x) > 0.01 or abs(q.y) > 0.01:
                self.state[6:10] = self._ros_quat_to_internal(q.x, q.y, q.z, q.w)

            if self.last_imu_stamp is None:
                self.last_imu_stamp = stamp_ns
                self.get_logger().info('EKF: First IMU received, using PX4 orientation directly')
                return

            dt = (stamp_ns - self.last_imu_stamp) * 1e-9
            self.last_imu_stamp = stamp_ns

            if dt <= 0 or dt > 0.5:
                if dt < -0.1:
                    self.get_logger().warn(
                        f'IMU time jump: dt={dt:.3f}s',
                        throttle_duration_sec=5.0
                    )
                return

            self._predict(
                msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                dt
            )
            self.imu_count += 1

    def _predict(self, ax, ay, az, gx, gy, gz, dt):
        pos = self.state[0:3].copy()
        vel = self.state[3:6].copy()
        q   = self.state[6:10].copy()           # Set by PX4 via IMU callback

        accel_body = np.array([ax, ay, az])     # PX4 handles bias internally
        R = self._quat_to_rotation(q)
        accel_world = R @ accel_body - self.gravity

        self.state[0:3] = pos + vel * dt + 0.5 * accel_world * dt * dt
        self.state[3:6] = vel + accel_world * dt
        drag = 0.99 if self._adaptive_vo_scale > 0.5 else 0.95
        self.state[3:6] *= drag

        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt

        self.P = F @ self.P @ F.T + self.Q * dt
        self._clamp_covariance()

        vel_mag = np.linalg.norm(self.state[3:6])
        if vel_mag > self.max_vel_sanity:
            self.state[3:6] *= (self.max_vel_sanity / vel_mag)
            self.vel_clamped_count += 1

    def _clamp_covariance(self):
        for i in range(15):
            if self.P[i, i] > self.p_diag_max:
                scale = math.sqrt(self.p_diag_max / self.P[i, i])
                self.P[i, :] *= scale
                self.P[:, i] *= scale
            if self.P[i, i] < 1e-10:
                self.P[i, i] = 1e-10

    def _compute_adaptive_gate(self):
        pos_unc = math.sqrt(
            max(0, self.P[0, 0]) + max(0, self.P[1, 1]) + max(0, self.P[2, 2])
        )
        adaptive = self.innov_gate_scale * pos_unc
        return np.clip(adaptive, self.innov_gate_min, self.max_pos_innov)

    def vo_callback(self, msg: Odometry):
        with self.lock:
            self.vo_count += 1
            self._recent_vo_accepted += 1
            if (self._recent_vo_accepted + self._recent_vo_rejected) >= 50:
                total = self._recent_vo_accepted + self._recent_vo_rejected
                reject_rate = self._recent_vo_rejected / total
                self._adaptive_vo_scale = max(0.2, 1.0 - reject_rate * 1.6)
                self._recent_vo_accepted = 0
                self._recent_vo_rejected = 0

            z_pos = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ])

            if self.vo_mode == 'incremental':
                if self._prev_vo_pos is None:
                    # First frame: just store anchor, init EKF
                    self._prev_vo_pos = z_pos.copy()
                    if not self._vo_initialized:
                        self._vo_initialized = True
                        self.initialized = True
                        self.last_vo_time = time.time()
                        self.get_logger().info(
                            f'EKF: First VO received (incremental mode), '
                            f'anchor at [{z_pos[0]:.2f}, {z_pos[1]:.2f}, {z_pos[2]:.2f}]'
                        )
                    return

                delta_vo = z_pos - self._prev_vo_pos
                self._prev_vo_pos = z_pos.copy()

                delta_norm = np.linalg.norm(delta_vo)
                if delta_norm > self.max_frame_delta:
                    delta_vo = delta_vo * (self.max_frame_delta / delta_norm)
                    delta_norm = self.max_frame_delta

                if delta_norm < 1e-6:
                    return

                z_adjusted = self.position.copy() + delta_vo
                y = z_adjusted - self.position  # = delta_vo (bounded)
                pos_innov_norm = np.linalg.norm(y)

            else:
                if not self._vo_initialized:
                    self.state[0:3] = z_pos.copy()
                    self.state[3:6] = 0.0
                    self.P[0:3, 0:3] = np.eye(3) * (self.vo_pos_noise ** 2)
                    self._vo_initialized = True
                    self.initialized = True
                    self.last_vo_time = time.time()
                    self.get_logger().info(
                        f'EKF: Initialized position from first VO: '
                        f'[{z_pos[0]:.2f}, {z_pos[1]:.2f}, {z_pos[2]:.2f}]'
                    )
                    return

                y = z_pos - self.position
                pos_innov_norm = np.linalg.norm(y)

            current_gate = self._compute_adaptive_gate()
            if pos_innov_norm > current_gate:
                self._consecutive_large_innov += 1
                self.abs_rejected_vo_count += 1
                self._recent_vo_accepted -= 1
                self._recent_vo_rejected += 1
                self.get_logger().warn(
                    f'VO ABS rejected: |innov|={pos_innov_norm:.2f}m '
                    f'> gate={current_gate:.1f}m '
                    f'(consec={self._consecutive_large_innov}, '
                    f'total={self.abs_rejected_vo_count}, '
                    f'trust={self._adaptive_vo_scale:.2f})',
                    throttle_duration_sec=2.0
                )
                if self._consecutive_large_innov >= self.innov_window_size:
                    self.state[3:6] *= 0.5  # Brake velocity
                    self._consecutive_large_innov = 0
                return

            self._consecutive_large_innov = 0

            H = np.zeros((3, 15))
            H[0:3, 0:3] = np.eye(3)

            R = self.R_pos.copy()
            if hasattr(msg.pose, 'covariance') and len(msg.pose.covariance) >= 36:
                cov = np.array(msg.pose.covariance).reshape(6, 6)
                pos_cov = cov[0:3, 0:3]
                diag = np.diag(pos_cov)
                if np.all(diag > 0) and np.all(diag < 100):
                    R = pos_cov

            S = H @ self.P @ H.T + R
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                self.get_logger().warn('EKF: Singular S, skipping')
                return

            mahal_dist_sq = float(y.T @ S_inv @ y)
            gate_scale = 1.0

            if mahal_dist_sq > self.mahal_hard_limit:
                self.rejected_vo_count += 1
                self._recent_vo_accepted -= 1
                self._recent_vo_rejected += 1
                self.get_logger().warn(
                    f'VO HARD rejected: Mahal²={mahal_dist_sq:.1f} '
                    f'|innov|={pos_innov_norm:.2f}m',
                    throttle_duration_sec=2.0
                )
                return
            elif mahal_dist_sq > self.mahal_threshold:
                if self.soft_gate_enabled:
                    overshoot = mahal_dist_sq - self.mahal_threshold
                    gate_range = self.mahal_hard_limit - self.mahal_threshold
                    gate_scale = max(0.05, 1.0 - overshoot / gate_range)
                    self.soft_gated_vo_count += 1
                else:
                    self.rejected_vo_count += 1
                    return

            K = self.P @ H.T @ S_inv
            effective_scale = gate_scale * self._adaptive_vo_scale
            K = K * effective_scale

            dx = K @ y                              # dx is 15-vector
            self.state[0:3] += dx[0:3]              # Position
            self.state[3:6] += dx[3:6]              # Velocity

            I_KH = np.eye(15) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
            self.P[0:6, 6:15] = 0.0
            self.P[6:15, 0:6] = 0.0
            self._clamp_covariance()
            self.initialized = True
            self._vo_accepted_count += 1
            self.last_vo_time = time.time()

    def publish_fused_state(self):
        with self.lock:
            if not self.initialized:
                return

            now = self.get_clock().now()

            odom = Odometry()
            odom.header.stamp = now.to_msg()
            odom.header.frame_id = 'map'
            odom.child_frame_id = 'base_link'
            odom.pose.pose.position.x = float(self.state[0])
            odom.pose.pose.position.y = float(self.state[1])
            odom.pose.pose.position.z = float(self.state[2])
            q = self.quaternion
            odom.pose.pose.orientation.x = float(q[1])
            odom.pose.pose.orientation.y = float(q[2])
            odom.pose.pose.orientation.z = float(q[3])
            odom.pose.pose.orientation.w = float(q[0])
            odom.twist.twist.linear.x = float(self.state[3])
            odom.twist.twist.linear.y = float(self.state[4])
            odom.twist.twist.linear.z = float(self.state[5])

            cov_36 = [0.0] * 36
            for i in range(3):
                for j in range(3):
                    cov_36[i * 6 + j] = float(self.P[i, j])
                    cov_36[(i+3) * 6 + (j+3)] = float(self.P[i+6, j+6])
            odom.pose.covariance = cov_36

            self.fused_odom_pub.publish(odom)

            pose = PoseStamped()
            pose.header = odom.header
            pose.pose = odom.pose.pose
            self.fused_pose_pub.publish(pose)

    def log_status(self):
        with self.lock:
            if self.initialized:
                pos = self.position
                vel_mag = np.linalg.norm(self.velocity)
                pos_uncertainty = math.sqrt(
                    max(0, self.P[0, 0]) + max(0, self.P[1, 1]) + max(0, self.P[2, 2])
                )
                roll, pitch, yaw = self._quat_to_euler(self.quaternion)
                self.get_logger().info(
                    f'EKF: pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] '
                    f'vel={vel_mag:.2f}m/s '
                    f'rpy=[{math.degrees(roll):.1f}°, {math.degrees(pitch):.1f}°, '
                    f'{math.degrees(yaw):.1f}°] '
                    f'unc={pos_uncertainty:.3f}m '
                    f'IMU={self.imu_count} VO={self.vo_count} '
                    f'accepted={self._vo_accepted_count} '
                    f'rej={self.rejected_vo_count} '
                    f'abs_rej={self.abs_rejected_vo_count} '
                    f'soft={self.soft_gated_vo_count} '
                    f'trust={self._adaptive_vo_scale:.2f} '
                    f'mode={self.vo_mode}'
                )
            else:
                self.get_logger().info('EKF: Waiting for first VO measurement...')

def main(args=None):
    rclpy.init(args=args)
    node = EKFFusion()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
