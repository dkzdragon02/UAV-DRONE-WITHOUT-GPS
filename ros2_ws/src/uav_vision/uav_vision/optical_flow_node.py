import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import TwistStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque
import time
import traceback

class OpticalFlowNode(Node):
    def __init__(self):
        super().__init__('optical_flow_node')
        self.declare_parameter('method', 'lucas_kanade')    # 'lucas_kanade' or 'farneback'
        self.declare_parameter('max_corners', 100)
        self.declare_parameter('quality_level', 0.01)
        self.declare_parameter('min_distance', 10)
        
        method = self.get_parameter('method').value
        
        self.bridge = CvBridge()
        self.prev_gray = None
        self.use_imu = self.declare_parameter('use_imu', True).value
        self.last_imu = None
        self.last_imu_time = None
        self.velocity_filter = deque(maxlen=10)             # Simple moving average filter

        self.image_timeout = float(self.declare_parameter('image_timeout', 2.0).value)
        self.imu_timeout = float(self.declare_parameter('imu_timeout', 2.0).value)
        self.last_image_time = None
    
        self.stats = {
            'frames_processed': 0,
            'avg_flow_magnitude': 0.0,
            'processing_times': deque(maxlen=100)
        }
        
        if method == 'lucas_kanade':
            self.use_lk = True
            self.lk_params = dict(
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            feature_params = dict(
                maxCorners=self.get_parameter('max_corners').value,
                qualityLevel=self.get_parameter('quality_level').value,
                minDistance=self.get_parameter('min_distance').value,
                blockSize=7
            )
            self.feature_params = feature_params
        else:
            self.use_lk = False
        
        self.twist_pub = self.create_publisher(TwistStamped, '/uav/optical_flow/twist', 10)
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        if self.use_imu:
            self.imu_sub = self.create_subscription(
                Imu,
                '/mavros/imu/data',
                self.imu_callback,
                qos_profile=qos_profile_sensor_data
            )
        
        self.get_logger().info(f'Optical Flow Node started (method: {method}, IMU: {self.use_imu})')
        self.watchdog_timer = self.create_timer(1.0, self.watchdog_check)
    
    def imu_callback(self, msg):
        self.last_imu = msg
        self.last_imu_time = time.time()
    
    def image_callback(self, msg):
        start_time = time.time()
        self.stats['frames_processed'] += 1
        self.last_image_time = start_time
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
            if len(cv_image.shape) == 3:
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv_image.copy()
            
            if self.prev_gray is None:
                self.prev_gray = gray
                return

            if self.use_lk:
                flow = self.compute_lucas_kanade(self.prev_gray, gray)
            else:
                flow = self.compute_farneback(self.prev_gray, gray)
            
            velocity = self.flow_to_velocity(flow)

            if self.use_imu and self.last_imu is not None:
                velocity = self.integrate_imu(velocity)

            self.velocity_filter.append(velocity)
            if len(self.velocity_filter) > 1:
                velocity = np.mean(self.velocity_filter, axis=0)

            flow_mag = np.linalg.norm(velocity)
            self.stats['avg_flow_magnitude'] = (
                self.stats['avg_flow_magnitude'] * (self.stats['frames_processed'] - 1) + flow_mag
            ) / self.stats['frames_processed']
            
            twist_msg = TwistStamped()
            twist_msg.header.stamp = msg.header.stamp
            twist_msg.header.frame_id = 'camera'
            twist_msg.twist.linear.x = float(velocity[0])
            twist_msg.twist.linear.y = float(velocity[1])
            twist_msg.twist.linear.z = float(velocity[2])
            
            self.twist_pub.publish(twist_msg)
            self.prev_gray = gray
            processing_time = time.time() - start_time
            self.stats['processing_times'].append(processing_time)
            
        except Exception as e:
            self.get_logger().error(f'Error processing optical flow: {e}\n{traceback.format_exc()}')

    def watchdog_check(self):
        now = time.time()
        if self.last_image_time is not None and (now - self.last_image_time) > self.image_timeout:
            self.get_logger().warn(f'No images received for {now - self.last_image_time:.1f}s')
        if self.use_imu and self.last_imu_time is not None and (now - self.last_imu_time) > self.imu_timeout:
            self.get_logger().warn(f'No IMU received for {now - self.last_imu_time:.1f}s')
    
    def compute_lucas_kanade(self, prev, curr):
        p0 = cv2.goodFeaturesToTrack(prev, mask=None, **self.feature_params)
        
        if p0 is None or len(p0) == 0:
            return None
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev, curr, p0, None, **self.lk_params)
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        if len(good_new) == 0:
            return None

        flow = good_new - good_old
        
        return flow
    
    def compute_farneback(self, prev, curr):
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        return flow
    
    def flow_to_velocity(self, flow):
        if flow is None:
            return np.array([0.0, 0.0, 0.0])
        
        if self.use_lk:
            mean_flow = np.mean(flow, axis=0)
            scale = 0.01  # pixels to m/s (cần calibrate)
            vx = mean_flow[0] * scale
            vy = mean_flow[1] * scale
        else:
            mean_flow = np.mean(flow, axis=(0, 1))
            scale = 0.01
            vx = mean_flow[0] * scale
            vy = mean_flow[1] * scale
        vz = 0.0
        
        return np.array([vx, vy, vz])
    
    def integrate_imu(self, flow_velocity: np.ndarray) -> np.ndarray:
        if self.last_imu is None:
            return flow_velocity
        
        imu_accel = np.array([
            self.last_imu.linear_acceleration.x,
            self.last_imu.linear_acceleration.y,
            self.last_imu.linear_acceleration.z
        ])
        
        # Simple integration (có thể cải thiện với Kalman filter)
        # Weight: 70% optical flow, 30% IMU
        imu_weight = 0.3
        flow_weight = 0.7
        
        # IMU velocity estimate (đơn giản hóa, cần tích phân đúng cách)
        # Chỉ sử dụng để điều chỉnh scale
        imu_magnitude = np.linalg.norm(imu_accel)
        flow_magnitude = np.linalg.norm(flow_velocity)
        
        if flow_magnitude > 0 and imu_magnitude > 0:
            scale_correction = min(1.5, max(0.5, imu_magnitude / (flow_magnitude + 0.1)))
            corrected_velocity = flow_velocity * (flow_weight + imu_weight * scale_correction)
        else:
            corrected_velocity = flow_velocity
        
        return corrected_velocity
    
    def get_statistics(self) -> dict:
        avg_time = np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0.0
        return {
            **self.stats,
            'avg_processing_time': avg_time
        }

def main(args=None):
    rclpy.init(args=args)
    node = OpticalFlowNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

