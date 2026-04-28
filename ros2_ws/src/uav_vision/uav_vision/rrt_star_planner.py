import numpy as np
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import random

@dataclass
class Node:
    position: np.ndarray
    parent: Optional['Node'] = None
    cost: float = 0.0
    children: List['Node'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

class RRTStarPlanner:  
    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        step_size: float = 0.5,
        goal_threshold: float = 0.5,
        max_iterations: int = 5000,
        rewire_radius: float = 1.0,
        obstacle_checker: Optional[callable] = None
    ):

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

        if len(start) != self.dimension or len(goal) != self.dimension:
            raise ValueError("Start and goal must match dimension")
        
        start_node = Node(position=np.array(start))
        self.nodes = [start_node]
        self.goal_node = None
        
        goal_array = np.array(goal)
        
        for iteration in range(self.max_iterations):
            if random.random() < 0.1:  # 10% chance to sample goal
                random_point = goal_array
            else:
                random_point = self._sample_random()
            
            nearest_node = self._nearest_node(random_point)
            new_node = self._steer(nearest_node, random_point)
            
            if self._collision_free(nearest_node.position, new_node.position):
                nearby_nodes = self._nearby_nodes(new_node.position, self.rewire_radius)
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
                
                new_node.parent = best_parent
                new_node.cost = best_cost
                best_parent.children.append(new_node)
                self.nodes.append(new_node)
                
                for node in nearby_nodes:
                    if node == best_parent:
                        continue
                    
                    new_cost = new_node.cost + self._distance(
                        new_node.position, node.position
                    )
                    if new_cost < node.cost and self._collision_free(
                        new_node.position, node.position
                    ):
                        if node.parent:
                            node.parent.children.remove(node)
                        node.parent = new_node
                        node.cost = new_cost
                        new_node.children.append(node)
                
                if self._distance(new_node.position, goal_array) < self.goal_threshold:
                    self.goal_node = new_node
                    return self._extract_path()
        
        return None 
    
    def _sample_random(self) -> np.ndarray:
        point = []
        for min_val, max_val in self.bounds:
            point.append(random.uniform(min_val, max_val))
        return np.array(point)
    
    def _nearest_node(self, point: np.ndarray) -> Node:
        min_dist = float('inf')
        nearest = None
        
        for node in self.nodes:
            dist = self._distance(node.position, point)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def _steer(self, from_node: Node, to_point: np.ndarray) -> Node:
        direction = to_point - from_node.position
        distance = np.linalg.norm(direction)
        
        if distance <= self.step_size:
            return Node(position=to_point.copy())
        
        new_position = from_node.position + (direction / distance) * self.step_size
        return Node(position=new_position)
    
    def _nearby_nodes(self, position: np.ndarray, radius: float) -> List[Node]:
        nearby = []
        for node in self.nodes:
            if self._distance(node.position, position) <= radius:
                nearby.append(node)
        return nearby
    
    def _collision_free(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        num_checks = max(3, int(np.linalg.norm(p2 - p1) / 0.1))
        
        for i in range(num_checks + 1):
            alpha = i / num_checks
            point = p1 + alpha * (p2 - p1)
            
            if self.obstacle_checker(point):
                return False
        
        return True
    
    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return np.linalg.norm(p2 - p1)
    
    def _extract_path(self) -> List[List[float]]:
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
        edges = []
        for node in self.nodes:
            if node.parent:
                edges.append((
                    node.parent.position.tolist(),
                    node.position.tolist()
                ))
        return edges

