#!/usr/bin/env python3
"""
Trajectory Optimizer
Optimizes paths for smoothness, safety, and efficiency
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.optimize import minimize
from scipy.interpolate import splrep, splev
import math


class TrajectoryOptimizer:
    """
    Trajectory optimizer for path smoothing and optimization.
    
    Features:
    - Path smoothing
    - Velocity optimization
    - Acceleration constraints
    - Obstacle avoidance
    """
    
    def __init__(
        self,
        max_velocity: float = 2.0,
        max_acceleration: float = 1.0,
        max_jerk: float = 0.5
    ):
        """
        Initialize trajectory optimizer.
        
        Args:
            max_velocity: Maximum velocity (m/s)
            max_acceleration: Maximum acceleration (m/s²)
            max_jerk: Maximum jerk (m/s³)
        """
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk
    
    def smooth_path(
        self,
        path: List[List[float]],
        smoothing_factor: float = 0.5
    ) -> List[List[float]]:
        """
        Smooth a path using simple moving average.
        
        Args:
            path: List of waypoints
            smoothing_factor: Smoothing factor (0-1)
            
        Returns:
            Smoothed path
        """
        if len(path) < 3:
            return path
        
        smoothed = [path[0]]  # Keep first point
        
        for i in range(1, len(path) - 1):
            prev = np.array(path[i - 1])
            curr = np.array(path[i])
            next_p = np.array(path[i + 1])
            
            # Weighted average
            smoothed_point = (
                (1 - smoothing_factor) * curr +
                (smoothing_factor / 2) * (prev + next_p)
            )
            smoothed.append(smoothed_point.tolist())
        
        smoothed.append(path[-1])  # Keep last point
        return smoothed
    
    def optimize_velocity_profile(
        self,
        path: List[List[float]],
        target_time: Optional[float] = None
    ) -> List[float]:
        """
        Optimize velocity profile along path.
        
        Args:
            path: List of waypoints
            target_time: Target time for traversal (None for minimum time)
            
        Returns:
            List of velocities for each segment
        """
        if len(path) < 2:
            return [0.0]
        
        # Calculate segment lengths
        segment_lengths = []
        for i in range(len(path) - 1):
            p1 = np.array(path[i])
            p2 = np.array(path[i + 1])
            length = np.linalg.norm(p2 - p1)
            segment_lengths.append(length)
        
        # Calculate minimum time for each segment
        min_times = []
        for length in segment_lengths:
            if length == 0:
                min_times.append(0.1)  # Small time for zero-length segments
            else:
                # Time to accelerate to max_velocity and decelerate
                t_accel = self.max_velocity / self.max_acceleration
                dist_accel = 0.5 * self.max_acceleration * t_accel ** 2
                
                if length <= 2 * dist_accel:
                    # Triangle profile
                    t = 2 * math.sqrt(length / self.max_acceleration)
                else:
                    # Trapezoid profile
                    t_const = (length - 2 * dist_accel) / self.max_velocity
                    t = 2 * t_accel + t_const
                
                min_times.append(t)
        
        total_min_time = sum(min_times)
        
        # If target time is less than minimum, use minimum
        if target_time is None or target_time < total_min_time:
            target_time = total_min_time
        
        # Scale times to meet target
        scale_factor = target_time / total_min_time
        times = [t * scale_factor for t in min_times]
        
        # Calculate velocities
        velocities = []
        for length, time in zip(segment_lengths, times):
            if time > 0:
                velocities.append(length / time)
            else:
                velocities.append(0.0)
        
        return velocities
    
    def add_acceleration_constraints(
        self,
        path: List[List[float]],
        velocities: List[float]
    ) -> List[float]:
        """
        Adjust velocities to satisfy acceleration constraints.
        
        Args:
            path: List of waypoints
            velocities: Initial velocities
            
        Returns:
            Adjusted velocities
        """
        if len(path) < 2 or len(velocities) < 1:
            return velocities
        
        adjusted = [velocities[0]]
        
        for i in range(1, len(velocities)):
            # Calculate distance
            p1 = np.array(path[i])
            p2 = np.array(path[i + 1])
            distance = np.linalg.norm(p2 - p1)
            
            if distance == 0:
                adjusted.append(0.0)
                continue
            
            # Maximum velocity change allowed
            prev_vel = adjusted[-1]
            max_vel_change = self.max_acceleration * (distance / prev_vel) if prev_vel > 0 else self.max_velocity
            
            # Constrain velocity
            new_vel = min(
                velocities[i],
                prev_vel + max_vel_change,
                self.max_velocity
            )
            new_vel = max(new_vel, 0.0)
            
            adjusted.append(new_vel)
        
        return adjusted
    
    def optimize_trajectory(
        self,
        path: List[List[float]],
        target_time: Optional[float] = None,
        smooth: bool = True
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Optimize complete trajectory.
        
        Args:
            path: Input path
            target_time: Target traversal time
            smooth: Whether to smooth path
            
        Returns:
            Tuple of (optimized_path, velocities)
        """
        # Smooth path if requested
        if smooth:
            optimized_path = self.smooth_path(path)
        else:
            optimized_path = path
        
        # Optimize velocity profile
        velocities = self.optimize_velocity_profile(optimized_path, target_time)
        
        # Apply acceleration constraints
        velocities = self.add_acceleration_constraints(optimized_path, velocities)
        
        return optimized_path, velocities
    
    def interpolate_path(
        self,
        path: List[List[float]],
        resolution: float = 0.1
    ) -> List[List[float]]:
        """
        Interpolate path to higher resolution.
        
        Args:
            path: Input path
            resolution: Desired resolution (meters)
            
        Returns:
            Interpolated path
        """
        if len(path) < 2:
            return path
        
        # Convert to numpy
        path_array = np.array(path)
        
        # Calculate cumulative distances
        distances = [0.0]
        for i in range(1, len(path)):
            dist = np.linalg.norm(path_array[i] - path_array[i - 1])
            distances.append(distances[-1] + dist)
        
        total_distance = distances[-1]
        
        # Generate interpolated points
        interpolated = []
        current_dist = 0.0
        
        while current_dist <= total_distance:
            # Find segment
            for i in range(len(distances) - 1):
                if distances[i] <= current_dist <= distances[i + 1]:
                    # Interpolate
                    alpha = (current_dist - distances[i]) / (
                        distances[i + 1] - distances[i]
                    ) if distances[i + 1] > distances[i] else 0.0
                    
                    point = path_array[i] + alpha * (path_array[i + 1] - path_array[i])
                    interpolated.append(point.tolist())
                    break
            
            current_dist += resolution
        
        # Ensure goal is included
        if interpolated[-1] != path[-1]:
            interpolated.append(path[-1])
        
        return interpolated

