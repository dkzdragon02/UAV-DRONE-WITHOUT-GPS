import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from sensor_msgs.msg import FluidPressure, Range
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from uav_vision.qos_profiles import mavros_qos as make_mavros_qos
import math
import time

class HeightSensorFusion(Node):
    def __init__(self):
        super().__init__('height_sensor_fusion')

        self.declare_parameter('baro_topic', '/mavros/imu/static_pressure')
        self.declare_parameter('rangefinder_topic', '/mavros/distance_sensor/rangefinder')
        self.declare_parameter('vo_odom_topic', '/uav/vision/odometry')
        self.declare_parameter('mavros_pose_topic', '/mavros/local_position/pose')
        self.declare_parameter('height_topic', '/uav/height_fused')
        self.declare_parameter('scale_topic', '/uav/scale_factor')
        self.declare_parameter('baro_alpha', 0.02)              # Complementary filter weight (low-pass)
        self.declare_parameter('rangefinder_max_range', 10.0)   # meters — trust range below this
        self.declare_parameter('rangefinder_min_range', 0.05)   # meters — ignore below this
        self.declare_parameter('scale_ema_alpha', 0.1)          # EMA smoothing for scale factor
        self.declare_parameter('reference_pressure', 101325.0)  # Pa — sea level pressure
        self.declare_parameter('publish_rate', 20.0)            # Hz

        baro_topic = self.get_parameter('baro_topic').value
        rangefinder_topic = self.get_parameter('rangefinder_topic').value
        vo_odom_topic = self.get_parameter('vo_odom_topic').value
        mavros_pose_topic = self.get_parameter('mavros_pose_topic').value
        height_topic = self.get_parameter('height_topic').value
        scale_topic = self.get_parameter('scale_topic').value

        self.baro_alpha = self.get_parameter('baro_alpha').value
        self.rangefinder_max = self.get_parameter('rangefinder_max_range').value
        self.rangefinder_min = self.get_parameter('rangefinder_min_range').value
        self.scale_ema_alpha = self.get_parameter('scale_ema_alpha').value
        self.reference_pressure = self.get_parameter('reference_pressure').value
        publish_rate = self.get_parameter('publish_rate').value

        self.baro_height = None             # Barometric altitude (meters)
        self.baro_reference = None          # Reference pressure at startup (Pa)
        self.rangefinder_height = None      # Rangefinder reading (meters)
        self.rangefinder_valid = False
        self.vo_height = 0.0                # VO estimated height
        self.mavros_height = 0.0            # PX4 local position height
        self.fused_height = 0.0             # Output: fused metric height
        self.scale_factor = 1.0             # Output: VO scale correction
        self.scale_initialized = False
        self.has_baro = False
        self.has_rangefinder = False
        self.has_vo = False
        self.last_baro_time = None
        self.last_rangefinder_time = None

        mavros_qos = make_mavros_qos()

        self.create_subscription(
            FluidPressure, baro_topic, self.baro_callback, mavros_qos
        )
        self.create_subscription(
            Range, rangefinder_topic, self.rangefinder_callback, mavros_qos
        )
        self.create_subscription(
            Odometry, vo_odom_topic, self.vo_callback, 10
        )
        self.create_subscription(
            PoseStamped, mavros_pose_topic, self.mavros_pose_callback, mavros_qos
        )

        self.height_pub = self.create_publisher(Float64, height_topic, 10)
        self.scale_pub = self.create_publisher(Float64, scale_topic, 10)
        self.create_timer(1.0 / publish_rate, self.publish_fused)
        self.create_timer(5.0, self.log_status)
        self.get_logger().info('Height Sensor Fusion started')
        self.get_logger().info(f'  Baro topic: {baro_topic}')
        self.get_logger().info(f'  Rangefinder topic: {rangefinder_topic}')
        self.get_logger().info(f'  VO topic: {vo_odom_topic}')

    def baro_callback(self, msg: FluidPressure):
        pressure = msg.fluid_pressure  # Pa

        if pressure <= 0:
            return

        if self.baro_reference is None:
            self.baro_reference = pressure
            self.get_logger().info(
                f'Baro reference set: {self.baro_reference:.1f} Pa'
            )

        self.baro_height = 44330.0 * (
            1.0 - (pressure / self.baro_reference) ** (1.0 / 5.255)
        )
        self.has_baro = True
        self.last_baro_time = time.time()

    def rangefinder_callback(self, msg: Range):
        distance = msg.range

        if (distance >= self.rangefinder_min and
                distance <= self.rangefinder_max and
                math.isfinite(distance)):
            self.rangefinder_height = distance
            self.rangefinder_valid = True
            self.has_rangefinder = True
            self.last_rangefinder_time = time.time()
        else:
            self.rangefinder_valid = False

    def vo_callback(self, msg: Odometry):
        self.vo_height = msg.pose.pose.position.z
        self.has_vo = True

    def mavros_pose_callback(self, msg: PoseStamped):
        self.mavros_height = msg.pose.position.z

    def _compute_fused_height(self):
        now = time.time()
        baro_stale = (self.last_baro_time is None or
                      now - self.last_baro_time > 2.0)
        rangefinder_stale = (self.last_rangefinder_time is None or
                             now - self.last_rangefinder_time > 1.0)

        if self.rangefinder_valid and not rangefinder_stale:
            if self.has_baro and not baro_stale:
                alpha = self.baro_alpha
                self.fused_height = (
                    alpha * self.baro_height +
                    (1.0 - alpha) * self.rangefinder_height
                )
            else:
                self.fused_height = self.rangefinder_height

        elif self.has_baro and not baro_stale:
            self.fused_height = self.baro_height

        else:
            self.fused_height = self.mavros_height

    def _compute_scale_factor(self):
        if abs(self.vo_height) < 0.3 or abs(self.fused_height) < 0.3:
            return  # Too close to ground, unreliable

        raw_scale = abs(self.fused_height) / abs(self.vo_height)

        if raw_scale < 0.01 or raw_scale > 100.0:
            return

        if not self.scale_initialized:
            self.scale_factor = raw_scale
            self.scale_initialized = True
        else:
            self.scale_factor = (
                self.scale_ema_alpha * raw_scale +
                (1.0 - self.scale_ema_alpha) * self.scale_factor
            )

    def publish_fused(self):
        self._compute_fused_height()
        self._compute_scale_factor()

        h_msg = Float64()
        h_msg.data = float(self.fused_height)
        self.height_pub.publish(h_msg)

        s_msg = Float64()
        s_msg.data = float(self.scale_factor)
        self.scale_pub.publish(s_msg)

    def log_status(self):
        sensors = []
        if self.has_baro:
            sensors.append(f'baro={self.baro_height:.2f}m')
        if self.has_rangefinder:
            sensors.append(
                f'rangefinder={self.rangefinder_height:.2f}m'
                f'({"OK" if self.rangefinder_valid else "STALE"})'
            )
        if self.has_vo:
            sensors.append(f'vo={self.vo_height:.2f}m')

        sensor_str = ', '.join(sensors) if sensors else 'no sensors'

        self.get_logger().info(
            f'Height: fused={self.fused_height:.2f}m, '
            f'scale={self.scale_factor:.3f}, '
            f'sources=[{sensor_str}]'
        )


def main(args=None):
    rclpy.init(args=args)
    node = HeightSensorFusion()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
