import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import csv
import os
import time
import math
import threading
from datetime import datetime
from typing import Optional

class DataLogger(Node):
    def __init__(self):
        super().__init__('data_logger')
        self.declare_parameter('log_directory', '~/uav_logs')
        self.declare_parameter('odom_topic', '/uav/vision/odometry')
        self.declare_parameter('fused_odom_topic', '/uav/fused_odometry')
        self.declare_parameter('imu_topic', '/mavros/imu/data')
        self.declare_parameter('state_topic', '/uav/state_machine/state')
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('enable_image_logging', False)
        self.declare_parameter('image_log_interval', 1.0)               # seconds between image saves
        self.declare_parameter('max_log_size_mb', 100)                  # Max size per CSV file
        self.declare_parameter('log_rate', 10.0)                        # Hz — how often to flush

        log_dir = self.get_parameter('log_directory').value
        odom_topic = self.get_parameter('odom_topic').value
        fused_odom_topic = self.get_parameter('fused_odom_topic').value
        imu_topic = self.get_parameter('imu_topic').value
        state_topic = self.get_parameter('state_topic').value
        camera_topic = self.get_parameter('camera_topic').value
        self.enable_image_logging = self.get_parameter('enable_image_logging').value
        self.image_log_interval = self.get_parameter('image_log_interval').value
        self.max_log_size_mb = self.get_parameter('max_log_size_mb').value

        log_dir = os.path.expanduser(log_dir)
        session_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(log_dir, f'session_{session_name}')
        os.makedirs(self.session_dir, exist_ok=True)

        if self.enable_image_logging:
            self.image_dir = os.path.join(self.session_dir, 'images')
            os.makedirs(self.image_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self._open_csv_files()
        self.current_state = "unknown"
        self.last_image_time = 0.0
        self.start_time = time.time()

        self.stats = {
            'odom_count': 0,
            'fused_odom_count': 0,
            'imu_count': 0,
            'state_count': 0,
            'image_count': 0,
            'bytes_written': 0,
        }

        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 10
        )
        self.fused_odom_sub = self.create_subscription(
            Odometry, fused_odom_topic, self.fused_odom_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, imu_topic, self.imu_callback,
            qos_profile=qos_profile_sensor_data
        )
        self.state_sub = self.create_subscription(
            String, state_topic, self.state_callback, 10
        )

        if self.enable_image_logging:
            self.camera_sub = self.create_subscription(
                Image, camera_topic, self.image_callback, 5
            )
        
        self.status_timer = self.create_timer(30.0, self.log_stats)
        self.get_logger().info(f'Data Logger started — session: {self.session_dir}')
        self.get_logger().info(f'Image logging: {"enabled" if self.enable_image_logging else "disabled"}')

    def _open_csv_files(self):
        self.odom_file = open(os.path.join(self.session_dir, 'odometry.csv'), 'w', newline='')
        self.odom_writer = csv.writer(self.odom_file)
        self.odom_writer.writerow([
            'timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw',
            'vx', 'vy', 'vz', 'wx', 'wy', 'wz',
            'cov_xx', 'cov_yy', 'cov_zz', 'source'
        ])

        self.fused_file = open(os.path.join(self.session_dir, 'fused_odometry.csv'), 'w', newline='')
        self.fused_writer = csv.writer(self.fused_file)
        self.fused_writer.writerow([
            'timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw',
            'vx', 'vy', 'vz', 'cov_xx', 'cov_yy', 'cov_zz'
        ])

        self.imu_file = open(os.path.join(self.session_dir, 'imu.csv'), 'w', newline='')
        self.imu_writer = csv.writer(self.imu_file)
        self.imu_writer.writerow([
            'timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz',
            'qx', 'qy', 'qz', 'qw'
        ])

        self.state_file = open(os.path.join(self.session_dir, 'state.csv'), 'w', newline='')
        self.state_writer = csv.writer(self.state_file)
        self.state_writer.writerow(['timestamp', 'state'])

        if self.enable_image_logging:
            self.image_index_file = open(os.path.join(self.session_dir, 'image_index.csv'), 'w', newline='')
            self.image_index_writer = csv.writer(self.image_index_file)
            self.image_index_writer.writerow(['timestamp', 'filename'])

    def odom_callback(self, msg: Odometry):
        with self.lock:
            t = time.time()
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            v = msg.twist.twist.linear
            w = msg.twist.twist.angular

            cov = msg.pose.covariance if len(msg.pose.covariance) >= 36 else [0]*36

            self.odom_writer.writerow([
                f'{t:.6f}', f'{p.x:.6f}', f'{p.y:.6f}', f'{p.z:.6f}',
                f'{q.x:.6f}', f'{q.y:.6f}', f'{q.z:.6f}', f'{q.w:.6f}',
                f'{v.x:.6f}', f'{v.y:.6f}', f'{v.z:.6f}',
                f'{w.x:.6f}', f'{w.y:.6f}', f'{w.z:.6f}',
                f'{cov[0]:.6f}', f'{cov[7]:.6f}', f'{cov[14]:.6f}',
                'vo'
            ])
            self.stats['odom_count'] += 1
            if self.stats['odom_count'] % 100 == 0:
                self.odom_file.flush()

    def fused_odom_callback(self, msg: Odometry):
        with self.lock:
            t = time.time()
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            v = msg.twist.twist.linear

            cov = msg.pose.covariance if len(msg.pose.covariance) >= 36 else [0]*36

            self.fused_writer.writerow([
                f'{t:.6f}', f'{p.x:.6f}', f'{p.y:.6f}', f'{p.z:.6f}',
                f'{q.x:.6f}', f'{q.y:.6f}', f'{q.z:.6f}', f'{q.w:.6f}',
                f'{v.x:.6f}', f'{v.y:.6f}', f'{v.z:.6f}',
                f'{cov[0]:.6f}', f'{cov[7]:.6f}', f'{cov[14]:.6f}'
            ])
            self.stats['fused_odom_count'] += 1

            if self.stats['fused_odom_count'] % 100 == 0:
                self.fused_file.flush()

    def imu_callback(self, msg: Imu):
        with self.lock:
            t = time.time()
            a = msg.linear_acceleration
            g = msg.angular_velocity
            q = msg.orientation

            self.imu_writer.writerow([
                f'{t:.6f}',
                f'{a.x:.6f}', f'{a.y:.6f}', f'{a.z:.6f}',
                f'{g.x:.6f}', f'{g.y:.6f}', f'{g.z:.6f}',
                f'{q.x:.6f}', f'{q.y:.6f}', f'{q.z:.6f}', f'{q.w:.6f}'
            ])
            self.stats['imu_count'] += 1

            if self.stats['imu_count'] % 500 == 0:
                self.imu_file.flush()

    def state_callback(self, msg: String):
        with self.lock:
            t = time.time()
            new_state = msg.data

            if new_state != self.current_state:
                self.state_writer.writerow([f'{t:.6f}', new_state])
                self.state_file.flush()
                self.current_state = new_state
                self.stats['state_count'] += 1
                self.get_logger().info(f'State logged: {new_state}')

    def image_callback(self, msg: Image):
        t = time.time()
        if t - self.last_image_time < self.image_log_interval:
            return

        self.last_image_time = t

        try:
            import cv2
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            filename = f'frame_{self.stats["image_count"]:06d}.jpg'
            filepath = os.path.join(self.image_dir, filename)
            cv2.imwrite(filepath, cv_image, [cv2.IMWRITE_JPEG_QUALITY, 85])

            with self.lock:
                self.image_index_writer.writerow([f'{t:.6f}', filename])
                self.stats['image_count'] += 1

                if self.stats['image_count'] % 10 == 0:
                    self.image_index_file.flush()

        except Exception as e:
            self.get_logger().error(f'Error saving image: {e}')

    def log_stats(self):
        elapsed = time.time() - self.start_time
        self.get_logger().info(
            f'DataLogger stats ({elapsed:.0f}s): '
            f'odom={self.stats["odom_count"]}, '
            f'fused={self.stats["fused_odom_count"]}, '
            f'imu={self.stats["imu_count"]}, '
            f'states={self.stats["state_count"]}, '
            f'images={self.stats["image_count"]}'
        )

    def _close_files(self):
        files = [self.odom_file, self.fused_file, self.imu_file, self.state_file]
        if self.enable_image_logging and hasattr(self, 'image_index_file'):
            files.append(self.image_index_file)

        for f in files:
            try:
                f.flush()
                f.close()
            except Exception:
                pass

    def _write_summary(self):
        elapsed = time.time() - self.start_time
        summary_path = os.path.join(self.session_dir, 'summary.txt')

        with open(summary_path, 'w') as f:
            f.write(f"UAV Data Logger Session Summary\n")
            f.write(f"{'='*40}\n")
            f.write(f"Session: {self.session_dir}\n")
            f.write(f"Duration: {elapsed:.1f} seconds\n")
            f.write(f"\nData Counts:\n")
            for key, value in self.stats.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nRates:\n")
            if elapsed > 0:
                f.write(f"  Odometry: {self.stats['odom_count']/elapsed:.1f} Hz\n")
                f.write(f"  IMU: {self.stats['imu_count']/elapsed:.1f} Hz\n")

        self.get_logger().info(f'Summary written to: {summary_path}')

    def destroy_node(self):
        self.get_logger().info('Shutting down Data Logger...')
        self._write_summary()
        self._close_files()
        self.log_stats()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DataLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
