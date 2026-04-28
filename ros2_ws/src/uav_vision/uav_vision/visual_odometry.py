import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import logging
from collections import deque
import time

logger = logging.getLogger(__name__)

class VisualOdometry:
    def __init__(self, 
                 feature_detector: str = "ORB",
                 max_features: int = 500,
                 min_match_count: int = 10,
                 ransac_threshold: float = 3.0,
                 scale_factor: float = 1.0,
                 max_drift_radius: float = 100.0,
                 max_frame_translation: float = 2.0,
                 dynamic_scale: bool = True):
        self.feature_detector = feature_detector
        self.max_features = max_features
        self.min_match_count = min_match_count
        self.ransac_threshold = ransac_threshold
        self.scale_factor = scale_factor
        self.max_drift_radius = max_drift_radius        # Soft position bound (meters)
        self.max_frame_translation = max_frame_translation  # Max plausible per-frame translation (m)
        self.dynamic_scale = dynamic_scale              # Allow external scale updates
        self._external_scale = None                     # Set by height sensor fusion
        self._velocity_damping = 1.0                    # Damping factor when near drift bound
        
        if feature_detector == "ORB":
            self.detector = cv2.ORB_create(nfeatures=max_features)
        elif feature_detector == "SIFT":
            self.detector = cv2.SIFT_create(nfeatures=max_features)
        elif feature_detector == "SURF":
            self.detector = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
        else:
            raise ValueError(f"Unsupported detector: {feature_detector}")
        
        if feature_detector == "ORB":
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.position = np.array([0.0, 0.0, 0.0])       # x, y, z (m)
        self.rotation = np.eye(3)                       # Rotation matrix
        self.translation = np.array([0.0, 0.0, 0.0])    # Translation vector
        self.camera_matrix = None
        self.dist_coeffs = None
        self.pose_covariance = np.eye(6) * 0.1          # Initial uncertainty
        self.scale_history = deque(maxlen=50)           # Lưu scale factors
        self.drift_compensation_enabled = True
        self.drift_threshold = 0.1                      # m/s drift threshold
        self._consecutive_clamps = 0
        self._max_consecutive_clamps = 30               # Start strong damping after this many clamps
        self._drift_warn_time = 0.0                     # Throttle warning logs
        self._covariance_inflated = False               # Signal for EKF
        
        self.stats = {
            'frames_processed': 0,
            'successful_frames': 0,
            'failed_frames': 0,
            'rejected_frames': 0,
            'avg_features': 0,
            'avg_matches': 0,
            'processing_times': deque(maxlen=100)
        }
        
        self._last_gray = None
        
        logger.info(f"Visual Odometry initialized with {feature_detector} detector")
    
    def set_camera_params(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
    
    def process_frame(self, frame: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        start_time = time.time()
        self.stats['frames_processed'] += 1
        
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.stats['avg_features'] = len(keypoints) if keypoints else 0
            return False, np.array([0.0, 0.0, 0.0]), np.eye(3), None
        
        if descriptors is None or len(descriptors) < self.min_match_count:
            logger.debug("Not enough features detected")
            self.stats['failed_frames'] += 1
            return False, np.array([0.0, 0.0, 0.0]), np.eye(3), None
        
        matches = self.matcher.match(self.prev_descriptors, descriptors)
        
        if len(matches) < self.min_match_count:
            logger.debug(f"Not enough matches: {len(matches)}")
            self.stats['failed_frames'] += 1
            return False, np.array([0.0, 0.0, 0.0]), np.eye(3), None
        
        self.stats['avg_features'] = (self.stats['avg_features'] * (self.stats['frames_processed'] - 1) + len(keypoints)) / self.stats['frames_processed']
        self.stats['avg_matches'] = (self.stats['avg_matches'] * (self.stats['frames_processed'] - 1) + len(matches)) / self.stats['frames_processed']
        
        src_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        if self.camera_matrix is not None:
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
            
            _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, self.camera_matrix, mask=mask)
            inlier_count = np.sum(mask) if mask is not None else len(matches)
            confidence = min(1.0, inlier_count / max(len(matches), 1))
            active_scale = self._get_active_scale()
            t = t.flatten() * active_scale
            covariance = self._estimate_covariance(R, t, confidence, inlier_count)
            
        else:
            H, mask = cv2.findHomography(
                src_pts, dst_pts,
                cv2.RANSAC,
                self.ransac_threshold
            )
            
            if H is None:
                logger.debug("Failed to compute Homography")
                self.stats['failed_frames'] += 1
                return False, np.array([0.0, 0.0, 0.0]), np.eye(3), None
            
            if self.camera_matrix is None:
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
            R = Rs[0] if len(Rs) > 0 else np.eye(3)
            t = Ts[0].flatten() if len(Ts) > 0 else np.array([0.0, 0.0, 0.0])
            t = t * self.scale_factor   # Áp dụng scale factor
            
            confidence = 0.7            # Homography ít chính xác hơn
            covariance = self._estimate_covariance(R, t, confidence, len(matches))
        
        if self.drift_compensation_enabled:
            t = self._apply_drift_compensation(t)
        
        t = t * self._velocity_damping
        
        t_world = self.rotation @ t
        t_norm = np.linalg.norm(t_world)
        if t_norm > self.max_frame_translation:
            logger.debug(
                f"Frame translation rejected: {t_norm:.3f}m > "
                f"{self.max_frame_translation:.1f}m limit"
            )
            self.stats['rejected_frames'] += 1
            t_world = t_world / t_norm * self.max_frame_translation
        
        self.translation = self.translation + t_world
        self.rotation = R @ self.rotation
        self.position = self.translation.copy()

        drift_distance = np.linalg.norm(self.position)
        if drift_distance > self.max_drift_radius:
            overshoot_ratio = drift_distance / self.max_drift_radius
            decay_factor = np.exp(-(overshoot_ratio - 1.0) * 2.0)
            target_distance = self.max_drift_radius * (1.0 - 0.05 * (1.0 - decay_factor))
            self.position = self.position / drift_distance * target_distance
            self.translation = self.position.copy()
            self._consecutive_clamps += 1
            
            now = time.time()
            if now - self._drift_warn_time > 5.0:
                logger.warning(
                    f"VO drift bound: {drift_distance:.1f}m > "
                    f"{self.max_drift_radius:.0f}m — "
                    f"softened to {target_distance:.1f}m "
                    f"(consecutive: {self._consecutive_clamps})"
                )
                self._drift_warn_time = now
            
            if self._consecutive_clamps >= self._max_consecutive_clamps:
                damping = max(0.1, 1.0 - (self._consecutive_clamps - self._max_consecutive_clamps) * 0.02)
                self._velocity_damping = damping
                self._covariance_inflated = True
                self.pose_covariance = np.eye(6) * 50.0
                
                if self._consecutive_clamps % self._max_consecutive_clamps == 0:
                    self.scale_history.clear()
                    logger.warning(
                        f"VO stuck at drift bound for {self._consecutive_clamps} frames — "
                        f"inflating covariance (damping={damping:.2f}), NOT resetting"
                    )
        else:
            self._consecutive_clamps = 0
            self._velocity_damping = 1.0
            self._covariance_inflated = False
        
        if covariance is not None:
            self.pose_covariance = self._update_covariance(covariance, R, t)
        
        self.prev_frame = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.stats['successful_frames'] += 1
        processing_time = time.time() - start_time
        self.stats['processing_times'].append(processing_time)
        
        return True, t, R, self.pose_covariance
    
    def get_position(self) -> np.ndarray:
        return self.position.copy()
    
    def get_rotation(self) -> np.ndarray:
        return self.rotation.copy()
    
    def _estimate_covariance(self, R: np.ndarray, t: np.ndarray, confidence: float, inlier_count: int) -> np.ndarray:
        pos_uncertainty = 0.1 / max(confidence, 0.1)                        # Position uncertainty (m)
        rot_uncertainty = 0.05 / max(confidence, 0.1)                       # Orientation uncertainty (rad)
        inlier_factor = max(1.0, 50.0 / max(inlier_count, 1))
        covariance = np.eye(6)
        covariance[0:3, 0:3] = np.eye(3) * pos_uncertainty * inlier_factor  # Position
        covariance[3:6, 3:6] = np.eye(3) * rot_uncertainty * inlier_factor  # Orientation
        
        return covariance
    
    def _update_covariance(self, new_cov: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        motion_uncertainty = np.eye(6) * 0.01  
        updated = self.pose_covariance + new_cov + motion_uncertainty
        max_cov = 100.0
        for i in range(6):
            if updated[i, i] > max_cov:
                scale = max_cov / updated[i, i]
                updated[i, :] *= scale
                updated[:, i] *= scale
        return updated
    
    def _apply_drift_compensation(self, t: np.ndarray) -> np.ndarray:
        t_norm = np.linalg.norm(t)
        if t_norm < 1e-9:
            return t

        if len(self.scale_history) >= 5:
            median_scale = float(np.median(list(self.scale_history)))
            if t_norm > 3.0 * max(median_scale, 0.01):
                logger.debug(
                    f"Drift: translation jump rejected "
                    f"(norm={t_norm:.4f}, 3×median={3*median_scale:.4f})"
                )
                t = t / t_norm * median_scale
                t_norm = median_scale

        self.scale_history.append(t_norm)

        if len(self.scale_history) >= 10:
            scales = np.array(list(self.scale_history))
            q1, q3 = np.percentile(scales, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            inliers = scales[(scales >= lower_bound) & (scales <= upper_bound)]

            if len(inliers) >= 3:
                adaptive_scale = float(np.median(inliers))
            else:
                adaptive_scale = float(np.median(scales))

            if t_norm > 0 and adaptive_scale > 0:
                compensation_factor = adaptive_scale / t_norm
                compensation_factor = np.clip(compensation_factor, 0.5, 2.0)
                if abs(compensation_factor - 1.0) > 0.02:
                    t = t * compensation_factor
                    logger.debug(
                        f"Drift compensation: factor={compensation_factor:.3f}, "
                        f"adaptive_scale={adaptive_scale:.4f}"
                    )

        return t
    
    def set_scale_factor(self, scale: float):
        if scale > 0.01 and scale < 100.0:
            self._external_scale = scale
            logger.debug(f"External scale factor updated: {scale:.4f}")
    
    def get_scale_factor(self) -> float:
        return self._get_active_scale()
    
    def _get_active_scale(self) -> float:
        if self.dynamic_scale and self._external_scale is not None:
            return self._external_scale
        return self.scale_factor
    
    def get_covariance(self) -> np.ndarray:
        return self.pose_covariance.copy()
    
    def get_statistics(self) -> Dict:
        avg_time = np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0.0
        return {
            **self.stats,
            'avg_processing_time': avg_time,
            'success_rate': self.stats['successful_frames'] / max(self.stats['frames_processed'], 1)
        }
    
    def reset(self):
        self.position = np.array([0.0, 0.0, 0.0])
        self.translation = np.array([0.0, 0.0, 0.0])
        self.rotation = np.eye(3)
        self.prev_frame = None
        self.pose_covariance = np.eye(6) * 0.1
        self.scale_history.clear()
        self._consecutive_clamps = 0
        self._velocity_damping = 1.0
        self._covariance_inflated = False
        self._drift_warn_time = 0.0
        self.stats = {
            'frames_processed': 0,
            'successful_frames': 0,
            'failed_frames': 0,
            'rejected_frames': 0,
            'avg_features': 0,
            'avg_matches': 0,
            'processing_times': deque(maxlen=100)
        }
        logger.info("Visual Odometry reset")

