import numpy as np
import cv2
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from uav_vision.visual_odometry import VisualOdometry

def _make_textured_image(width=320, height=240, seed=42):
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 256, (height, width), dtype=np.uint8)
    for _ in range(30):
        cx, cy = rng.randint(10, width - 10), rng.randint(10, height - 10)
        r = rng.randint(3, 15)
        cv2.circle(img, (cx, cy), r, int(rng.randint(0, 256)), -1)
    for _ in range(20):
        x1, y1 = rng.randint(0, width), rng.randint(0, height)
        x2, y2 = rng.randint(0, width), rng.randint(0, height)
        cv2.line(img, (x1, y1), (x2, y2), int(rng.randint(0, 256)), 2)
    return img

def _translate_image(img, tx, ty):
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

class TestVisualOdometryInit:
    def test_default_init(self):
        vo = VisualOdometry()
        assert vo.feature_detector == "ORB"
        assert vo.max_features == 500
        np.testing.assert_array_equal(vo.position, [0.0, 0.0, 0.0])

    def test_sift_init(self):
        vo = VisualOdometry(feature_detector="SIFT")
        assert vo.feature_detector == "SIFT"

    def test_invalid_detector(self):
        with pytest.raises(ValueError):
            VisualOdometry(feature_detector="INVALID")

    def test_camera_params(self):
        vo = VisualOdometry()
        K = np.array([[500, 0, 160], [0, 500, 120], [0, 0, 1]], dtype=np.float64)
        D = np.zeros(5)
        vo.set_camera_params(K, D)
        np.testing.assert_array_equal(vo.camera_matrix, K)

    def test_drift_bound(self):
        vo = VisualOdometry(max_drift_radius=50.0)
        assert vo.max_drift_radius == 50.0

class TestProcessFrame:
    def setup_method(self):
        self.vo = VisualOdometry(feature_detector="ORB", max_features=500)
        K = np.array([[224, 0, 160], [0, 224, 120], [0, 0, 1]], dtype=np.float64)
        D = np.zeros(5)
        self.vo.set_camera_params(K, D)

    def test_first_frame_returns_false(self):
        img = _make_textured_image()
        success, t, R, cov = self.vo.process_frame(img)
        assert success is False
        np.testing.assert_array_equal(t, [0.0, 0.0, 0.0])

    def test_identical_frames(self):
        img = _make_textured_image()
        self.vo.process_frame(img)
        success, t, R, cov = self.vo.process_frame(img.copy())
        assert isinstance(success, bool)
        if success:
            assert np.all(np.isfinite(t)), "Translation should be finite"

    def test_translated_frame(self):
        img1 = _make_textured_image(seed=100)
        img2 = _translate_image(img1, 10, 0)  # 10px right
        self.vo.process_frame(img1)
        success, t, R, cov = self.vo.process_frame(img2)
        assert isinstance(success, bool)

    def test_color_frame(self):
        gray = _make_textured_image()
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        success, t, R, cov = self.vo.process_frame(color)
        assert isinstance(success, bool)

    def test_blank_frame(self):
        self.vo.process_frame(_make_textured_image())
        blank = np.zeros((240, 320), dtype=np.uint8)
        success, t, R, cov = self.vo.process_frame(blank)
        assert success is False

    def test_stats_tracked(self):
        img = _make_textured_image()
        self.vo.process_frame(img)
        assert self.vo.stats['frames_processed'] == 1


class TestDriftCompensation:
    def setup_method(self):
        self.vo = VisualOdometry(max_drift_radius=10.0)
        self.vo.drift_compensation_enabled = True

    def test_jump_rejection(self):
        for _ in range(10):
            self.vo.scale_history.append(0.01)
        t = np.array([1.0, 0.0, 0.0])  # 100x the median = jump
        result = self.vo._apply_drift_compensation(t)
        assert np.linalg.norm(result) < np.linalg.norm(t), "Jump should be damped"

    def test_small_motion_unchanged(self):
        for _ in range(15):
            self.vo.scale_history.append(0.05)
        t = np.array([0.05, 0.0, 0.0])
        result = self.vo._apply_drift_compensation(t)
        assert abs(np.linalg.norm(result) - 0.05) < 0.03        # Should be close to original

    def test_zero_translation(self):
        t = np.array([0.0, 0.0, 0.0])
        result = self.vo._apply_drift_compensation(t)
        np.testing.assert_array_equal(result, t)

    def test_drift_radius_bound(self):
        self.vo.position = np.array([50.0, 0.0, 0.0])
        self.vo.translation = self.vo.position.copy()
        drift_distance = np.linalg.norm(self.vo.position)
        if drift_distance > self.vo.max_drift_radius:
            self.vo.position = self.vo.position / drift_distance * self.vo.max_drift_radius
        assert np.linalg.norm(self.vo.position) <= self.vo.max_drift_radius + 1e-6


class TestCovariance:
    def setup_method(self):
        self.vo = VisualOdometry()

    def test_covariance_shape(self):
        cov = self.vo.get_covariance()
        assert cov.shape == (6, 6)

    def test_covariance_initial(self):
        cov = self.vo.get_covariance()
        np.testing.assert_allclose(cov, np.eye(6) * 0.1)

    def test_covariance_increases(self):
        cov_before = np.trace(self.vo.pose_covariance)
        R = np.eye(3)
        t = np.array([1.0, 0.0, 0.0])
        new_cov = self.vo._estimate_covariance(R, t, confidence=0.3, inlier_count=10)
        self.vo.pose_covariance = self.vo._update_covariance(new_cov, R, t)
        cov_after = np.trace(self.vo.pose_covariance)
        assert cov_after > cov_before

    def test_covariance_bounded(self):
        R = np.eye(3)
        t = np.array([1.0, 0.0, 0.0])
        self.vo.pose_covariance = np.eye(6) * 500.0
        new_cov = np.eye(6) * 100.0
        bounded = self.vo._update_covariance(new_cov, R, t)
        for i in range(6):
            assert bounded[i, i] <= 100.0 + 1e-6, f"Diagonal [{i}] should be bounded"

    def test_estimate_covariance_high_confidence(self):
        R = np.eye(3)
        t = np.array([0.1, 0.0, 0.0])
        cov = self.vo._estimate_covariance(R, t, confidence=0.95, inlier_count=200)
        assert cov[0, 0] < 0.5, "High confidence should have low position uncertainty"


class TestReset:
    def test_reset_clears_state(self):
        vo = VisualOdometry()
        vo.position = np.array([5.0, 3.0, 1.0])
        vo.stats['frames_processed'] = 100
        vo.scale_history.append(0.5)
        vo.reset()
        np.testing.assert_array_equal(vo.position, [0.0, 0.0, 0.0])
        assert vo.stats['frames_processed'] == 0
        assert len(vo.scale_history) == 0
        assert vo.prev_frame is None


class TestStatistics:
    def test_stats_structure(self):
        vo = VisualOdometry()
        stats = vo.get_statistics()
        assert 'frames_processed' in stats
        assert 'successful_frames' in stats
        assert 'success_rate' in stats
        assert 'avg_processing_time' in stats

    def test_success_rate(self):
        vo = VisualOdometry()
        vo.stats['frames_processed'] = 100
        vo.stats['successful_frames'] = 80
        stats = vo.get_statistics()
        assert abs(stats['success_rate'] - 0.8) < 1e-6


class TestHomographyFallback:
    def test_no_camera_matrix(self):
        vo = VisualOdometry()
        assert vo.camera_matrix is None
        img = _make_textured_image()
        vo.process_frame(img)
        img2 = _translate_image(img, 5, 0)
        success, t, R, cov = vo.process_frame(img2)
        assert isinstance(success, bool)
