"""
Module Visual Odometry sử dụng camera
Tính toán vị trí và hướng từ hình ảnh liên tiếp
Tối ưu với covariance estimation, scale recovery, và drift compensation
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import logging
from collections import deque
import time

logger = logging.getLogger(__name__)


class VisualOdometry:
    """Visual Odometry sử dụng feature matching"""
    
    def __init__(self, 
                 feature_detector: str = "ORB",
                 max_features: int = 500,
                 min_match_count: int = 10,
                 ransac_threshold: float = 3.0,
                 scale_factor: float = 1.0):
        """
        Khởi tạo Visual Odometry
        
        Args:
            feature_detector: Loại detector ("ORB", "SIFT", "SURF")
            max_features: Số lượng features tối đa
            min_match_count: Số matches tối thiểu để tính toán
            ransac_threshold: Ngưỡng RANSAC
            scale_factor: Tỷ lệ thực tế (m/pixel)
        """
        self.feature_detector = feature_detector
        self.max_features = max_features
        self.min_match_count = min_match_count
        self.ransac_threshold = ransac_threshold
        self.scale_factor = scale_factor
        
        # Khởi tạo feature detector
        if feature_detector == "ORB":
            self.detector = cv2.ORB_create(nfeatures=max_features)
        elif feature_detector == "SIFT":
            self.detector = cv2.SIFT_create(nfeatures=max_features)
        elif feature_detector == "SURF":
            self.detector = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
        else:
            raise ValueError(f"Unsupported detector: {feature_detector}")
        
        # Matcher
        if feature_detector == "ORB":
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        # Lưu frame trước
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        
        # Vị trí và hướng hiện tại (tích lũy)
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z (m)
        self.rotation = np.eye(3)  # Rotation matrix
        self.translation = np.array([0.0, 0.0, 0.0])  # Translation vector
        
        # Camera intrinsics (cần calibrate cho camera cụ thể)
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Covariance estimation (6x6: position + orientation)
        self.pose_covariance = np.eye(6) * 0.1  # Initial uncertainty
        
        # Scale recovery và drift compensation
        self.scale_history = deque(maxlen=50)  # Lưu scale factors
        self.drift_compensation_enabled = True
        self.drift_threshold = 0.1  # m/s drift threshold
        
        # Statistics tracking
        self.stats = {
            'frames_processed': 0,
            'successful_frames': 0,
            'failed_frames': 0,
            'avg_features': 0,
            'avg_matches': 0,
            'processing_times': deque(maxlen=100)
        }
        
        # Performance optimization: cache grayscale conversion
        self._last_gray = None
        
        logger.info(f"Visual Odometry initialized with {feature_detector} detector")
    
    def set_camera_params(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """
        Thiết lập tham số camera
        
        Args:
            camera_matrix: Ma trận camera intrinsics (3x3)
            dist_coeffs: Hệ số distortion
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
    
    def process_frame(self, frame: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Xử lý frame mới và tính toán chuyển động (tối ưu)
        
        Args:
            frame: Frame ảnh (grayscale hoặc color)
            
        Returns:
            (success, translation, rotation_matrix, covariance)
        """
        start_time = time.time()
        self.stats['frames_processed'] += 1
        
        # Chuyển sang grayscale nếu cần (cached)
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Detect features
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        if self.prev_frame is None:
            # Frame đầu tiên
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.stats['avg_features'] = len(keypoints) if keypoints else 0
            return False, np.array([0.0, 0.0, 0.0]), np.eye(3), None
        
        # Match features
        if descriptors is None or len(descriptors) < self.min_match_count:
            logger.debug("Not enough features detected")
            self.stats['failed_frames'] += 1
            return False, np.array([0.0, 0.0, 0.0]), np.eye(3), None
        
        matches = self.matcher.match(self.prev_descriptors, descriptors)
        
        if len(matches) < self.min_match_count:
            logger.debug(f"Not enough matches: {len(matches)}")
            self.stats['failed_frames'] += 1
            return False, np.array([0.0, 0.0, 0.0]), np.eye(3), None
        
        # Update statistics
        self.stats['avg_features'] = (self.stats['avg_features'] * (self.stats['frames_processed'] - 1) + len(keypoints)) / self.stats['frames_processed']
        self.stats['avg_matches'] = (self.stats['avg_matches'] * (self.stats['frames_processed'] - 1) + len(matches)) / self.stats['frames_processed']
        
        # Lấy điểm tương ứng
        src_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Tính toán Essential Matrix hoặc Homography
        if self.camera_matrix is not None:
            # Sử dụng Essential Matrix (chính xác hơn cho camera calibrated)
            E, mask = cv2.findEssentialMat(
                src_pts, dst_pts,
                self.camera_matrix,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=self.ransac_threshold
            )
            
            if E is None:
                logger.debug("Failed to compute Essential Matrix")
                self.stats['failed_frames'] += 1
                return False, np.array([0.0, 0.0, 0.0]), np.eye(3), None
            
            # Recover pose
            _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, self.camera_matrix, mask=mask)
            
            # Estimate covariance từ số lượng inliers
            inlier_count = np.sum(mask) if mask is not None else len(matches)
            confidence = min(1.0, inlier_count / max(len(matches), 1))
            covariance = self._estimate_covariance(R, t, confidence, inlier_count)
            
        else:
            # Sử dụng Homography (không cần camera calibration)
            H, mask = cv2.findHomography(
                src_pts, dst_pts,
                cv2.RANSAC,
                self.ransac_threshold
            )
            
            if H is None:
                logger.debug("Failed to compute Homography")
                self.stats['failed_frames'] += 1
                return False, np.array([0.0, 0.0, 0.0]), np.eye(3), None
            
            # Decompose Homography để lấy rotation và translation
            # (phương pháp đơn giản hóa, cần camera matrix)
            # Nếu không có camera matrix, sử dụng ước lượng
            if self.camera_matrix is None:
                # Tạo camera matrix ước lượng
                h, w = gray.shape[:2]
                fx = w * 0.7
                fy = h * 0.7
                cx = w / 2.0
                cy = h / 2.0
                cam_matrix = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=np.float32)
            else:
                cam_matrix = self.camera_matrix
            num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, cam_matrix)
            # Chọn solution phù hợp (thường là solution đầu tiên)
            R = Rs[0] if len(Rs) > 0 else np.eye(3)
            t = Ts[0].flatten() if len(Ts) > 0 else np.array([0.0, 0.0, 0.0])
            t = t * self.scale_factor  # Áp dụng scale factor
            
            # Estimate covariance cho homography (thấp hơn essential matrix)
            confidence = 0.7  # Homography ít chính xác hơn
            covariance = self._estimate_covariance(R, t, confidence, len(matches))
        
        # Scale recovery và drift compensation
        if self.drift_compensation_enabled:
            t = self._apply_drift_compensation(t)
        
        # Cập nhật vị trí tích lũy
        self.translation = self.translation + self.rotation @ t
        self.rotation = R @ self.rotation
        self.position = self.translation.copy()
        
        # Update covariance
        if covariance is not None:
            self.pose_covariance = self._update_covariance(covariance, R, t)
        
        # Lưu frame hiện tại
        self.prev_frame = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        
        # Update statistics
        self.stats['successful_frames'] += 1
        processing_time = time.time() - start_time
        self.stats['processing_times'].append(processing_time)
        
        return True, t, R, self.pose_covariance
    
    def get_position(self) -> np.ndarray:
        """Lấy vị trí hiện tại (tích lũy)"""
        return self.position.copy()
    
    def get_rotation(self) -> np.ndarray:
        """Lấy rotation matrix"""
        return self.rotation.copy()
    
    def _estimate_covariance(self, R: np.ndarray, t: np.ndarray, confidence: float, inlier_count: int) -> np.ndarray:
        """
        Ước lượng covariance từ confidence và số lượng inliers
        
        Args:
            R: Rotation matrix
            t: Translation vector
            confidence: Confidence level (0-1)
            inlier_count: Số lượng inliers
            
        Returns:
            6x6 covariance matrix (position + orientation)
        """
        # Base uncertainty
        pos_uncertainty = 0.1 / max(confidence, 0.1)  # Position uncertainty (m)
        rot_uncertainty = 0.05 / max(confidence, 0.1)  # Orientation uncertainty (rad)
        
        # Scale với số lượng inliers
        inlier_factor = max(1.0, 50.0 / max(inlier_count, 1))
        
        covariance = np.eye(6)
        covariance[0:3, 0:3] = np.eye(3) * pos_uncertainty * inlier_factor  # Position
        covariance[3:6, 3:6] = np.eye(3) * rot_uncertainty * inlier_factor  # Orientation
        
        return covariance
    
    def _update_covariance(self, new_cov: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Cập nhật covariance tích lũy với motion uncertainty
        
        Args:
            new_cov: Covariance mới từ frame hiện tại
            R: Rotation matrix
            t: Translation vector
            
        Returns:
            Updated 6x6 covariance matrix
        """
        # Simple propagation (có thể cải thiện với EKF)
        motion_uncertainty = np.eye(6) * 0.01  # Small motion uncertainty
        return self.pose_covariance + new_cov + motion_uncertainty
    
    def _apply_drift_compensation(self, t: np.ndarray) -> np.ndarray:
        """
        Áp dụng drift compensation dựa trên lịch sử scale
        
        Args:
            t: Translation vector
            
        Returns:
            Compensated translation vector
        """
        t_norm = np.linalg.norm(t)
        if t_norm > 0:
            self.scale_history.append(t_norm)
            
            # Nếu có đủ lịch sử, kiểm tra drift
            if len(self.scale_history) > 10:
                recent_scales = list(self.scale_history)[-10:]
                avg_scale = np.mean(recent_scales)
                std_scale = np.std(recent_scales)
                
                # Nếu scale thay đổi đột ngột, có thể là drift
                if std_scale > self.drift_threshold:
                    # Giảm scale để giảm drift
                    compensation_factor = 0.95
                    t = t * compensation_factor
                    logger.debug(f"Applied drift compensation: factor={compensation_factor}")
        
        return t
    
    def get_covariance(self) -> np.ndarray:
        """Lấy covariance hiện tại"""
        return self.pose_covariance.copy()
    
    def get_statistics(self) -> Dict:
        """Lấy thống kê xử lý"""
        avg_time = np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0.0
        return {
            **self.stats,
            'avg_processing_time': avg_time,
            'success_rate': self.stats['successful_frames'] / max(self.stats['frames_processed'], 1)
        }
    
    def reset(self):
        """Reset vị trí về gốc"""
        self.position = np.array([0.0, 0.0, 0.0])
        self.translation = np.array([0.0, 0.0, 0.0])
        self.rotation = np.eye(3)
        self.prev_frame = None
        self.pose_covariance = np.eye(6) * 0.1
        self.scale_history.clear()
        self.stats = {
            'frames_processed': 0,
            'successful_frames': 0,
            'failed_frames': 0,
            'avg_features': 0,
            'avg_matches': 0,
            'processing_times': deque(maxlen=100)
        }
        logger.info("Visual Odometry reset")

