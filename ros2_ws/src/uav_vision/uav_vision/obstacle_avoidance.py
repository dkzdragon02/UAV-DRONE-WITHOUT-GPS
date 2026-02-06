#!/usr/bin/env python3
"""
Advanced Obstacle Avoidance
Dynamic obstacle avoidance với potential fields và reactive behaviors
"""

import numpy as np
from typing import List, Tuple, Optional
import math


class ObstacleAvoidance:
    """Advanced obstacle avoidance"""
    
    def __init__(self, repulsive_gain: float = 1.0, attractive_gain: float = 1.0):
        """
        Args:
            repulsive_gain: Gain cho repulsive force
            attractive_gain: Gain cho attractive force
        """
        self.repulsive_gain = repulsive_gain
        self.attractive_gain = attractive_gain
        self.safety_distance = 0.5  # meters
        self.max_repulsive_distance = 2.0  # meters
    
    def compute_potential_field(self, current_pos: np.ndarray, goal_pos: np.ndarray,
                               obstacles: List[np.ndarray]) -> np.ndarray:
        """
        Compute potential field force
        
        Args:
            current_pos: Current position [x, y]
            goal_pos: Goal position [x, y]
            obstacles: List of obstacle positions [[x, y], ...]
        
        Returns:
            Force vector [fx, fy]
        """
        # Attractive force to goal
        attractive_force = self.compute_attractive_force(current_pos, goal_pos)
        
        # Repulsive force from obstacles
        repulsive_force = np.array([0.0, 0.0])
        for obstacle in obstacles:
            repulsive = self.compute_repulsive_force(current_pos, obstacle)
            repulsive_force += repulsive
        
        # Total force
        total_force = self.attractive_gain * attractive_force + self.repulsive_gain * repulsive_force
        
        return total_force
    
    def compute_attractive_force(self, current_pos: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
        """Compute attractive force to goal"""
        direction = goal_pos - current_pos
        distance = np.linalg.norm(direction)
        
        if distance < 0.01:
            return np.array([0.0, 0.0])
        
        # Normalize
        direction = direction / distance
        
        # Force magnitude (linear)
        magnitude = min(distance, 1.0)  # Limit max force
        
        return direction * magnitude
    
    def compute_repulsive_force(self, current_pos: np.ndarray, obstacle_pos: np.ndarray) -> np.ndarray:
        """Compute repulsive force from obstacle"""
        direction = current_pos - obstacle_pos
        distance = np.linalg.norm(direction)
        
        if distance > self.max_repulsive_distance:
            return np.array([0.0, 0.0])
        
        if distance < 0.01:
            # Too close, strong repulsion in random direction
            return np.array([1.0, 0.0]) * 10.0
        
        # Normalize
        direction = direction / distance
        
        # Force magnitude (inverse square)
        if distance < self.safety_distance:
            magnitude = 1.0 / (distance ** 2)
        else:
            magnitude = 1.0 / (distance ** 2) * (self.max_repulsive_distance - distance) / self.max_repulsive_distance
        
        return direction * magnitude
    
    def avoid_obstacles(self, current_pos: np.ndarray, goal_pos: np.ndarray,
                       obstacles: List[np.ndarray], max_velocity: float = 1.0) -> np.ndarray:
        """
        Compute velocity command to avoid obstacles
        
        Args:
            current_pos: Current position [x, y]
            goal_pos: Goal position [x, y]
            obstacles: List of obstacle positions
            max_velocity: Maximum velocity
        
        Returns:
            Velocity command [vx, vy]
        """
        force = self.compute_potential_field(current_pos, goal_pos, obstacles)
        
        # Convert force to velocity
        velocity = force * max_velocity
        
        # Limit velocity
        speed = np.linalg.norm(velocity)
        if speed > max_velocity:
            velocity = velocity / speed * max_velocity
        
        return velocity
    
    def check_collision(self, pos: np.ndarray, obstacles: List[np.ndarray],
                       obstacle_radius: float = 0.3) -> bool:
        """Check if position collides with obstacles"""
        for obstacle in obstacles:
            distance = np.linalg.norm(pos - obstacle)
            if distance < obstacle_radius:
                return True
        return False
    
    def find_safe_path(self, start: np.ndarray, goal: np.ndarray,
                      obstacles: List[np.ndarray], num_waypoints: int = 10) -> List[np.ndarray]:
        """
        Find safe path using potential fields
        
        Args:
            start: Start position [x, y]
            goal: Goal position [x, y]
            obstacles: List of obstacles
            num_waypoints: Number of waypoints to generate
        
        Returns:
            List of waypoints
        """
        waypoints = []
        current = start.copy()
        step_size = 0.2  # meters
        
        for i in range(num_waypoints):
            # Check if reached goal
            distance_to_goal = np.linalg.norm(current - goal)
            if distance_to_goal < 0.5:
                waypoints.append(goal.copy())
                break
            
            # Compute force
            force = self.compute_potential_field(current, goal, obstacles)
            
            # Normalize and scale
            force_norm = np.linalg.norm(force)
            if force_norm > 0.01:
                force = force / force_norm * step_size
            
            # Update position
            next_pos = current + force
            
            # Check collision
            if not self.check_collision(next_pos, obstacles):
                current = next_pos
                waypoints.append(current.copy())
            else:
                # Try to go around obstacle
                # Find direction perpendicular to obstacle
                if obstacles:
                    closest_obstacle = min(obstacles, key=lambda o: np.linalg.norm(current - o))
                    direction = current - closest_obstacle
                    direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0.01 else np.array([1.0, 0.0])
                    
                    # Perpendicular
                    perp = np.array([-direction[1], direction[0]])
                    next_pos = current + perp * step_size
                    
                    if not self.check_collision(next_pos, obstacles):
                        current = next_pos
                        waypoints.append(current.copy())
        
        return waypoints

