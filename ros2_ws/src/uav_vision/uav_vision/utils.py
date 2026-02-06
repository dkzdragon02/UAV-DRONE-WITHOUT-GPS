#!/usr/bin/env python3
"""
Utility functions for UAV Vision System
Common helper functions and utilities
"""

import numpy as np
from typing import Optional, Tuple, List
import time
from dataclasses import dataclass
import math


@dataclass
class Timer:
    """Simple timer for performance measurement"""
    start_time: float = 0.0
    
    def start(self):
        """Start the timer"""
        self.start_time = time.time()
    
    def elapsed(self) -> float:
        """Get elapsed time in seconds"""
        return time.time() - self.start_time
    
    def reset(self):
        """Reset the timer"""
        self.start_time = time.time()


def quaternion_to_euler(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).
    
    Args:
        qx, qy, qz, qw: Quaternion components
        
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return (roll, pitch, yaw)


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """
    Convert Euler angles to quaternion.
    
    Args:
        roll, pitch, yaw: Euler angles in radians
        
    Returns:
        Tuple of (qx, qy, qz, qw)
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return (qx, qy, qz, qw)


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-pi, pi].
    
    Args:
        angle: Angle in radians
        
    Returns:
        Normalized angle
    """
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def distance_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calculate 2D Euclidean distance.
    
    Args:
        p1: First point (x, y)
        p2: Second point (x, y)
        
    Returns:
        Distance
    """
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def distance_3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    """
    Calculate 3D Euclidean distance.
    
    Args:
        p1: First point (x, y, z)
        p2: Second point (x, y, z)
        
    Returns:
        Distance
    """
    return math.sqrt(
        (p2[0] - p1[0])**2 + 
        (p2[1] - p1[1])**2 + 
        (p2[2] - p1[2])**2
    )


def wrap_value(value: float, min_val: float, max_val: float) -> float:
    """
    Wrap value to range [min_val, max_val].
    
    Args:
        value: Value to wrap
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Wrapped value
    """
    range_size = max_val - min_val
    if range_size <= 0:
        return min_val
    
    while value < min_val:
        value += range_size
    while value > max_val:
        value -= range_size
    
    return value


def clamp_value(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp value to range [min_val, max_val].
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def calculate_covariance_ellipse(
    covariance: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate covariance ellipse parameters.
    
    Args:
        covariance: 2x2 covariance matrix
        confidence: Confidence level (default 0.95)
        
    Returns:
        Tuple of (major_axis, minor_axis, angle)
    """
    # Eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(covariance)
    
    # Sort by eigenvalue
    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # Chi-squared value for confidence level
    from scipy.stats import chi2
    chi2_val = chi2.ppf(confidence, df=2)
    
    # Ellipse axes
    major_axis = 2 * math.sqrt(chi2_val * eigenvals[0])
    minor_axis = 2 * math.sqrt(chi2_val * eigenvals[1])
    
    # Angle
    angle = math.atan2(eigenvecs[1, 0], eigenvecs[0, 0])
    
    return (major_axis, minor_axis, angle)


def rate_limiter(max_rate: float):
    """
    Create a rate limiter function.
    
    Args:
        max_rate: Maximum rate (calls per second)
        
    Returns:
        Rate limiter function
    """
    min_interval = 1.0 / max_rate
    last_call = [0.0]
    
    def limiter():
        current_time = time.time()
        elapsed = current_time - last_call[0]
        
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        
        last_call[0] = time.time()
        return True
    
    return limiter


def moving_average(values: List[float], window_size: int) -> List[float]:
    """
    Calculate moving average.
    
    Args:
        values: List of values
        window_size: Window size
        
    Returns:
        List of moving averages
    """
    if len(values) < window_size:
        return values
    
    result = []
    for i in range(len(values)):
        start = max(0, i - window_size + 1)
        window = values[start:i + 1]
        result.append(sum(window) / len(window))
    
    return result


def exponential_smoothing(
    current: float,
    previous: float,
    alpha: float
) -> float:
    """
    Apply exponential smoothing.
    
    Args:
        current: Current value
        previous: Previous smoothed value
        alpha: Smoothing factor (0-1)
        
    Returns:
        Smoothed value
    """
    return alpha * current + (1 - alpha) * previous

