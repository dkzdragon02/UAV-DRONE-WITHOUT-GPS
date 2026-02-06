#!/usr/bin/env python3
"""
RRT* Path Planner
Rapidly-exploring Random Tree Star algorithm for path planning
"""

import numpy as np
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import random


@dataclass
class Node:
    """RRT* node"""
    position: np.ndarray
    parent: Optional['Node'] = None
    cost: float = 0.0
    children: List['Node'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


class RRTStarPlanner:
    """
    RRT* (Rapidly-exploring Random Tree Star) path planner.
    
    Features:
    - Optimal path finding
    - Rewiring for optimization
    - Collision checking
    - Configurable parameters
    """
    
    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        step_size: float = 0.5,
        goal_threshold: float = 0.5,
        max_iterations: int = 5000,
        rewire_radius: float = 1.0,
        obstacle_checker: Optional[callable] = None
    ):
        """
        Initialize RRT* planner.
        
        Args:
            bounds: Bounds for each dimension [(min, max), ...]
            step_size: Step size for tree expansion
            goal_threshold: Distance threshold to consider goal reached
            max_iterations: Maximum iterations
            rewire_radius: Radius for rewiring optimization
            obstacle_checker: Function to check collisions (point -> bool)
        """
        self.bounds = bounds
        self.dimension = len(bounds)
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        self.max_iterations = max_iterations
        self.rewire_radius = rewire_radius
        self.obstacle_checker = obstacle_checker or (lambda p: False)
        
        self.nodes: List[Node] = []
        self.goal_node: Optional[Node] = None
    
    def plan(
        self,
        start: List[float],
        goal: List[float]
    ) -> Optional[List[List[float]]]:
        """
        Plan a path from start to goal.
        
        Args:
            start: Start position
            goal: Goal position
            
        Returns:
            List of waypoints or None if planning failed
        """
        # Validate inputs
        if len(start) != self.dimension or len(goal) != self.dimension:
            raise ValueError("Start and goal must match dimension")
        
        # Initialize tree with start node
        start_node = Node(position=np.array(start))
        self.nodes = [start_node]
        self.goal_node = None
        
        goal_array = np.array(goal)
        
        # Main planning loop
        for iteration in range(self.max_iterations):
            # Sample random point
            if random.random() < 0.1:  # 10% chance to sample goal
                random_point = goal_array
            else:
                random_point = self._sample_random()
            
            # Find nearest node
            nearest_node = self._nearest_node(random_point)
            
            # Extend towards random point
            new_node = self._steer(nearest_node, random_point)
            
            # Check collision
            if self._collision_free(nearest_node.position, new_node.position):
                # Find nearby nodes for rewiring
                nearby_nodes = self._nearby_nodes(new_node.position, self.rewire_radius)
                
                # Choose best parent
                best_parent = nearest_node
                best_cost = nearest_node.cost + self._distance(
                    nearest_node.position, new_node.position
                )
                
                for node in nearby_nodes:
                    if node == nearest_node:
                        continue
                    
                    cost = node.cost + self._distance(node.position, new_node.position)
                    if cost < best_cost and self._collision_free(
                        node.position, new_node.position
                    ):
                        best_parent = node
                        best_cost = cost
                
                # Add node to tree
                new_node.parent = best_parent
                new_node.cost = best_cost
                best_parent.children.append(new_node)
                self.nodes.append(new_node)
                
                # Rewire nearby nodes
                for node in nearby_nodes:
                    if node == best_parent:
                        continue
                    
                    new_cost = new_node.cost + self._distance(
                        new_node.position, node.position
                    )
                    if new_cost < node.cost and self._collision_free(
                        new_node.position, node.position
                    ):
                        # Rewire
                        if node.parent:
                            node.parent.children.remove(node)
                        node.parent = new_node
                        node.cost = new_cost
                        new_node.children.append(node)
                
                # Check if goal reached
                if self._distance(new_node.position, goal_array) < self.goal_threshold:
                    self.goal_node = new_node
                    return self._extract_path()
        
        return None  # Planning failed
    
    def _sample_random(self) -> np.ndarray:
        """Sample a random point within bounds"""
        point = []
        for min_val, max_val in self.bounds:
            point.append(random.uniform(min_val, max_val))
        return np.array(point)
    
    def _nearest_node(self, point: np.ndarray) -> Node:
        """Find nearest node to point"""
        min_dist = float('inf')
        nearest = None
        
        for node in self.nodes:
            dist = self._distance(node.position, point)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def _steer(self, from_node: Node, to_point: np.ndarray) -> Node:
        """Steer from node towards point"""
        direction = to_point - from_node.position
        distance = np.linalg.norm(direction)
        
        if distance <= self.step_size:
            return Node(position=to_point.copy())
        
        # Step towards point
        new_position = from_node.position + (direction / distance) * self.step_size
        return Node(position=new_position)
    
    def _nearby_nodes(self, position: np.ndarray, radius: float) -> List[Node]:
        """Find nodes within radius"""
        nearby = []
        for node in self.nodes:
            if self._distance(node.position, position) <= radius:
                nearby.append(node)
        return nearby
    
    def _collision_free(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Check if path between two points is collision-free"""
        # Check intermediate points
        num_checks = max(3, int(np.linalg.norm(p2 - p1) / 0.1))
        
        for i in range(num_checks + 1):
            alpha = i / num_checks
            point = p1 + alpha * (p2 - p1)
            
            if self.obstacle_checker(point):
                return False
        
        return True
    
    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate Euclidean distance"""
        return np.linalg.norm(p2 - p1)
    
    def _extract_path(self) -> List[List[float]]:
        """Extract path from goal to start"""
        if not self.goal_node:
            return []
        
        path = []
        current = self.goal_node
        
        while current:
            path.append(current.position.tolist())
            current = current.parent
        
        path.reverse()
        return path
    
    def get_tree(self) -> List[Tuple[List[float], List[float]]]:
        """
        Get tree edges for visualization.
        
        Returns:
            List of (parent, child) position pairs
        """
        edges = []
        for node in self.nodes:
            if node.parent:
                edges.append((
                    node.parent.position.tolist(),
                    node.position.tolist()
                ))
        return edges

