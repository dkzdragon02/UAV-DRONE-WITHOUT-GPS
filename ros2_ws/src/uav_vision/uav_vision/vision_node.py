#!/usr/bin/env python3
"""
ROS2 Node cho Vision Processing
Hỗ trợ ORB-SLAM3, VINS, và Visual Odometry
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys
import os
import yaml
import traceback
from tf2_ros import TransformBroadcaster

# Import local modules
from uav_vision.visual_odometry import VisualOdometry
from uav_vision.camera_handler import CameraHandler


class VisionNode(Node):
    """ROS2 Node xử lý vision cho UAV"""
    
    def __init__(self):
        super().__init__('vision_node')
        
        # Parameters
        self.declare_parameter('use_orb_slam', False)
        self.declare_parameter('use_vins', False)
        self.declare_parameter('use_visual_odometry', True)
        self.declare_parameter('camera_width', 640)
        self.declare_parameter('camera_height', 480)
        self.declare_parameter('camera_fps', 30)
        self.declare_parameter('camera_calibration', '')
        self.declare_parameter('frame_id', 'vision_odom')
        self.declare_parameter('child_frame_id', 'base_link')
        
        self.use_orb_slam = self.get_parameter('use_orb_slam').value
        self.use_vins = self.get_parameter('use_vins').value
        self.use_visual_odometry = self.get_parameter('use_visual_odometry').value
        self.frame_id = self.get_parameter('frame_id').value
        self.child_frame_id = self.get_parameter('child_frame_id').value
        self.camera_calib_path = self.get_parameter('camera_calibration').value
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Initialize vision systems
        if self.use_visual_odometry:
            self.visual_odometry = VisualOdometry(feature_detector="ORB", max_features=500)
            self.camera = CameraHandler(
                width=self.get_parameter('camera_width').value,
                height=self.get_parameter('camera_height').value,
                fps=self.get_parameter('camera_fps').value,
                calibration_file=self.camera_calib_path if self.camera_calib_path else None
            )
            # Apply calibration params to VO if available
            cam_matrix = self.camera.get_camera_matrix()
            dist_coeffs = self.camera.get_distortion_coeffs()
            if cam_matrix is not None:
                self.visual_odometry.set_camera_params(cam_matrix, dist_coeffs)
            self.get_logger().info("Visual Odometry initialized")
        
        if self.use_orb_slam:
            # TODO: Initialize ORB-SLAM3 wrapper
            self.get_logger().warn("ORB-SLAM3 not yet implemented")
        
        if self.use_vins:
            # TODO: Initialize OpenVINS wrapper
            self.get_logger().warn("OpenVINS not yet implemented")
        
        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/uav/vision/pose', 10)
        self.odom_pub = self.create_publisher(Odometry, '/uav/vision/odometry', 10)
        self.twist_pub = self.create_publisher(TwistStamped, '/uav/vision/twist', 10)
        self.status_pub = self.create_publisher(Odometry, '/uav/vision/status', 1)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )
        
        # Timer cho camera capture (nếu không có ROS image topic)
        self.timer = self.create_timer(1.0 / self.get_parameter('camera_fps').value, self.timer_callback)
        self.health_timer = self.create_timer(1.0, self.health_check)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Health tracking
        self.last_image_time = None
        self.last_success_time = None
        
        self.get_logger().info('Vision Node started')
    
    def image_callback(self, msg):
        """Callback khi nhận image từ ROS topic"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.last_image_time = self.get_clock().now()
            self.process_image(cv_image)
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    def imu_callback(self, msg):
        """Callback khi nhận IMU data"""
        # Lưu IMU data để sử dụng trong sensor fusion
        self.last_imu = msg
    
    def timer_callback(self):
        """Timer callback để đọc camera trực tiếp"""
        if hasattr(self, 'camera') and self.camera.is_initialized:
            frame = self.camera.read_frame()
            if frame is not None:
                self.last_image_time = self.get_clock().now()
                self.process_image(frame)
    
    def process_image(self, image):
        """Xử lý image và publish pose/odometry (tối ưu)"""
        if self.use_visual_odometry and hasattr(self, 'visual_odometry'):
            try:
                result = self.visual_odometry.process_frame(image)
                if len(result) == 4:
                    success, translation, rotation, covariance = result
                else:
                    # Backward compatibility
                    success, translation, rotation = result
                    covariance = None
                
                if success:
                    now_msg = self.get_clock().now().to_msg()
                    self.last_success_time = self.get_clock().now()
                    position = self.visual_odometry.get_position()
                    
                    # Ensure position is 1D array and extract scalar values
                    position = np.asarray(position).flatten()
                    
                    # Helper function to safely convert numpy array element to float
                    def to_float(val):
                        if isinstance(val, np.ndarray):
                            if val.size == 1:
                                return float(val.item())
                            else:
                                return float(val.flatten()[0])
                        return float(val)
                    
                    # Publish PoseStamped
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = now_msg
                    pose_msg.header.frame_id = self.frame_id
                    pose_msg.pose.position.x = to_float(position[0])
                    pose_msg.pose.position.y = to_float(position[1])
                    pose_msg.pose.position.z = to_float(position[2])
                    
                    # Convert rotation matrix to quaternion
                    quat = self.rotation_matrix_to_quaternion(rotation)
                    quat = np.asarray(quat).flatten()
                    pose_msg.pose.orientation.w = to_float(quat[0])
                    pose_msg.pose.orientation.x = to_float(quat[1])
                    pose_msg.pose.orientation.y = to_float(quat[2])
                    pose_msg.pose.orientation.z = to_float(quat[3])
                    
                    self.pose_pub.publish(pose_msg)
                    
                    # Publish Odometry với covariance
                    odom_msg = Odometry()
                    odom_msg.header.stamp = pose_msg.header.stamp
                    odom_msg.header.frame_id = self.frame_id
                    odom_msg.child_frame_id = self.child_frame_id
                    odom_msg.pose.pose = pose_msg.pose
                    
                    # Thêm covariance nếu có
                    if covariance is not None:
                        # Convert 6x6 covariance (pos+rot) to 36-element array
                        # Format: [xx, xy, xz, xroll, xpitch, xyaw, yx, yy, ...]
                        cov_flat = covariance.flatten()
                        if len(cov_flat) >= 36:
                            odom_msg.pose.covariance = [float(x) for x in cov_flat[:36]]
                        else:
                            # Fallback: chỉ position covariance
                            pos_cov = covariance[:3, :3].flatten()
                            odom_msg.pose.covariance[:9] = [float(x) for x in pos_cov[:9]]
                    
                    self.odom_pub.publish(odom_msg)
            except Exception as e:
                self.get_logger().error(f'Error in process_image: {e}\n{traceback.format_exc()}')

                # Publish TF
                t = TransformStamped()
                t.header.stamp = pose_msg.header.stamp
                t.header.frame_id = self.frame_id
                t.child_frame_id = self.child_frame_id
                t.transform.translation.x = pose_msg.pose.position.x
                t.transform.translation.y = pose_msg.pose.position.y
                t.transform.translation.z = pose_msg.pose.position.z
                t.transform.rotation = pose_msg.pose.orientation
                self.tf_broadcaster.sendTransform(t)
    
    def rotation_matrix_to_quaternion(self, R):
        """Chuyển rotation matrix sang quaternion"""
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        
        return np.array([w, x, y, z])

    def health_check(self):
        """Đánh giá tình trạng luồng ảnh/VO (nâng cao)"""
        now = self.get_clock().now()
        status_ok = True
        warnings = []
        
        # Kiểm tra image stream
        if self.last_image_time:
            image_age = (now - self.last_image_time).nanoseconds * 1e-9
            if image_age > 2.0:
                warnings.append(f"No images for {image_age:.1f}s")
                status_ok = False
        
        # Kiểm tra VO success rate
        if self.last_success_time:
            vo_age = (now - self.last_success_time).nanoseconds * 1e-9
            if vo_age > 5.0:
                warnings.append(f"No VO updates for {vo_age:.1f}s")
                status_ok = False
        
        # Lấy statistics từ VO nếu có
        stats_info = ""
        if hasattr(self, 'visual_odometry') and self.use_visual_odometry:
            stats = self.visual_odometry.get_statistics()
            stats_info = f" | Success: {stats['success_rate']*100:.1f}% | Avg features: {stats['avg_features']:.0f}"
            if stats['avg_processing_time'] > 0:
                stats_info += f" | Avg time: {stats['avg_processing_time']*1000:.1f}ms"
        
        # Log warnings nếu có
        if warnings:
            self.get_logger().warn("; ".join(warnings) + stats_info)
        elif stats_info:
            self.get_logger().debug(f"Health OK{stats_info}")
        
        # Publish status
        status_msg = Odometry()
        status_msg.header.stamp = now.to_msg()
        status_msg.header.frame_id = self.frame_id
        if not status_ok:
            status_msg.pose.covariance[0] = 999.0  # flag high covariance
        self.status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'camera'):
            node.camera.release()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

