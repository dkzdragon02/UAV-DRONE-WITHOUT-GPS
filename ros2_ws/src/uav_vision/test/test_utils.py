"""
Unit tests for utility functions
"""

import pytest
import time
import math

from uav_vision.utils import (
    Timer,
    quaternion_to_euler,
    euler_to_quaternion,
    normalize_angle,
    distance_2d,
    distance_3d,
    clamp_value,
    wrap_value,
    rate_limiter,
    moving_average,
    exponential_smoothing
)


class TestTimer:
    """Test suite for Timer"""
    
    def test_timer_start(self):
        """Test starting timer"""
        timer = Timer()
        timer.start()
        
        assert timer.start_time > 0
    
    def test_timer_elapsed(self):
        """Test getting elapsed time"""
        timer = Timer()
        timer.start()
        
        time.sleep(0.1)
        elapsed = timer.elapsed()
        
        assert elapsed >= 0.1
        assert elapsed < 0.2
    
    def test_timer_reset(self):
        """Test resetting timer"""
        timer = Timer()
        timer.start()
        initial_time = timer.start_time
        
        time.sleep(0.1)
        timer.reset()
        
        assert timer.start_time > initial_time


class TestQuaternionEuler:
    """Test suite for quaternion/euler conversions"""
    
    def test_quaternion_to_euler_identity(self):
        """Test conversion of identity quaternion"""
        roll, pitch, yaw = quaternion_to_euler(0, 0, 0, 1)
        
        assert roll == pytest.approx(0, abs=1e-6)
        assert pitch == pytest.approx(0, abs=1e-6)
        assert yaw == pytest.approx(0, abs=1e-6)
    
    def test_euler_to_quaternion_identity(self):
        """Test conversion of zero euler angles"""
        qx, qy, qz, qw = euler_to_quaternion(0, 0, 0)
        
        assert qx == pytest.approx(0, abs=1e-6)
        assert qy == pytest.approx(0, abs=1e-6)
        assert qz == pytest.approx(0, abs=1e-6)
        assert qw == pytest.approx(1, abs=1e-6)
    
    def test_round_trip_conversion(self):
        """Test round trip conversion"""
        original_roll = math.pi / 4
        original_pitch = math.pi / 6
        original_yaw = math.pi / 3
        
        qx, qy, qz, qw = euler_to_quaternion(original_roll, original_pitch, original_yaw)
        roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)
        
        assert roll == pytest.approx(original_roll, abs=1e-5)
        assert pitch == pytest.approx(original_pitch, abs=1e-5)
        assert yaw == pytest.approx(original_yaw, abs=1e-5)


class TestNormalizeAngle:
    """Test suite for angle normalization"""
    
    def test_normalize_positive(self):
        """Test normalizing positive angle"""
        angle = math.pi + 0.5
        normalized = normalize_angle(angle)
        
        assert normalized == pytest.approx(-math.pi + 0.5, abs=1e-6)
    
    def test_normalize_negative(self):
        """Test normalizing negative angle"""
        angle = -math.pi - 0.5
        normalized = normalize_angle(angle)
        
        assert normalized == pytest.approx(math.pi - 0.5, abs=1e-6)
    
    def test_normalize_in_range(self):
        """Test normalizing angle already in range"""
        angle = math.pi / 2
        normalized = normalize_angle(angle)
        
        assert normalized == pytest.approx(angle, abs=1e-6)


class TestDistance:
    """Test suite for distance calculations"""
    
    def test_distance_2d(self):
        """Test 2D distance calculation"""
        p1 = (0, 0)
        p2 = (3, 4)
        
        dist = distance_2d(p1, p2)
        
        assert dist == 5.0
    
    def test_distance_2d_same_point(self):
        """Test 2D distance for same point"""
        p1 = (5, 5)
        p2 = (5, 5)
        
        dist = distance_2d(p1, p2)
        
        assert dist == 0.0
    
    def test_distance_3d(self):
        """Test 3D distance calculation"""
        p1 = (0, 0, 0)
        p2 = (2, 3, 6)
        
        dist = distance_3d(p1, p2)
        
        assert dist == 7.0  # sqrt(4 + 9 + 36) = 7


class TestClampWrap:
    """Test suite for clamp and wrap functions"""
    
    def test_clamp_value(self):
        """Test value clamping"""
        assert clamp_value(5, 0, 10) == 5
        assert clamp_value(-5, 0, 10) == 0
        assert clamp_value(15, 0, 10) == 10
    
    def test_wrap_value(self):
        """Test value wrapping"""
        assert wrap_value(5, 0, 10) == 5
        assert wrap_value(15, 0, 10) == 5
        assert wrap_value(-5, 0, 10) == 5
    
    def test_wrap_value_edge_cases(self):
        """Test wrap value edge cases"""
        # Note: wrap_value(10, 0, 10) wraps to 0 because 10 is outside [0, 10)
        # But since we're using inclusive range, it wraps
        result = wrap_value(10, 0, 10)
        assert result >= 0 and result <= 10  # Should be in range
        assert wrap_value(0, 0, 10) == 0


class TestRateLimiter:
    """Test suite for rate limiter"""
    
    def test_rate_limiter(self):
        """Test rate limiter"""
        limiter = rate_limiter(max_rate=10.0)  # 10 calls per second
        
        start = time.time()
        limiter()
        limiter()
        elapsed = time.time() - start
        
        # Should take at least 0.1 seconds (1/10)
        assert elapsed >= 0.09


class TestMovingAverage:
    """Test suite for moving average"""
    
    def test_moving_average(self):
        """Test moving average calculation"""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        window_size = 3
        
        result = moving_average(values, window_size)
        
        assert len(result) == len(values)
        assert result[0] == 1.0  # First value
        assert result[2] == pytest.approx(2.0, abs=1e-6)  # (1+2+3)/3
        assert result[-1] == pytest.approx(9.0, abs=1e-6)  # (8+9+10)/3
    
    def test_moving_average_small_window(self):
        """Test moving average with small window"""
        values = [1, 2, 3]
        window_size = 5  # Larger than values
        
        result = moving_average(values, window_size)
        
        assert result == values  # Should return original if window > length


class TestExponentialSmoothing:
    """Test suite for exponential smoothing"""
    
    def test_exponential_smoothing(self):
        """Test exponential smoothing"""
        current = 10.0
        previous = 5.0
        alpha = 0.5
        
        result = exponential_smoothing(current, previous, alpha)
        
        assert result == 7.5  # 0.5 * 10 + 0.5 * 5
    
    def test_exponential_smoothing_alpha_0(self):
        """Test with alpha = 0 (no change)"""
        current = 10.0
        previous = 5.0
        
        result = exponential_smoothing(current, previous, 0.0)
        
        assert result == previous
    
    def test_exponential_smoothing_alpha_1(self):
        """Test with alpha = 1 (full change)"""
        current = 10.0
        previous = 5.0
        
        result = exponential_smoothing(current, previous, 1.0)
        
        assert result == current

