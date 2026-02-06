#!/usr/bin/env python3
"""
Graph Optimization cho SLAM
Cải thiện loop closure với pose graph optimization
"""

import numpy as np
from typing import List, Tuple, Dict
import math


class PoseGraph:
    """Pose Graph cho SLAM optimization"""
    
    def __init__(self):
        self.nodes = []  # List of poses [x, y, yaw]
        self.edges = []  # List of constraints [(i, j, relative_pose, covariance)]
        self.loop_closures = []  # List of loop closure edges
    
    def add_node(self, pose: np.ndarray):
        """Add pose node"""
        self.nodes.append(pose.copy())
        return len(self.nodes) - 1
    
    def add_edge(self, i: int, j: int, relative_pose: np.ndarray, covariance: np.ndarray = None):
        """Add edge constraint"""
        if covariance is None:
            covariance = np.eye(3) * 0.1
        
        self.edges.append({
            'from': i,
            'to': j,
            'relative_pose': relative_pose.copy(),
            'covariance': covariance.copy()
        })
    
    def add_loop_closure(self, i: int, j: int, relative_pose: np.ndarray, covariance: np.ndarray = None):
        """Add loop closure constraint"""
        if covariance is None:
            covariance = np.eye(3) * 0.05  # Loop closures are more certain
        
        self.loop_closures.append({
            'from': i,
            'to': j,
            'relative_pose': relative_pose.copy(),
            'covariance': covariance.copy()
        })
    
    def optimize(self, iterations: int = 10) -> List[np.ndarray]:
        """Optimize pose graph using Gauss-Newton"""
        if len(self.nodes) < 2:
            return self.nodes.copy()
        
        optimized_poses = [pose.copy() for pose in self.nodes]
        
        for iteration in range(iterations):
            # Compute residuals and Jacobians
            residuals = []
            jacobians = []
            
            # Odometry edges
            for edge in self.edges:
                i, j = edge['from'], edge['to']
                if j >= len(optimized_poses):
                    continue
                
                residual, jacobian = self.compute_residual(
                    optimized_poses[i],
                    optimized_poses[j],
                    edge['relative_pose'],
                    edge['covariance']
                )
                residuals.append(residual)
                jacobians.append(jacobian)
            
            # Loop closure edges
            for edge in self.loop_closures:
                i, j = edge['from'], edge['to']
                if j >= len(optimized_poses):
                    continue
                
                residual, jacobian = self.compute_residual(
                    optimized_poses[i],
                    optimized_poses[j],
                    edge['relative_pose'],
                    edge['covariance']
                )
                residuals.append(residual)
                jacobians.append(jacobian)
            
            if not residuals:
                break
            
            # Build system
            H = np.zeros((len(optimized_poses) * 3, len(optimized_poses) * 3))
            b = np.zeros(len(optimized_poses) * 3)
            
            for residual, jacobian in zip(residuals, jacobians):
                # Simplified: add to system
                # In full implementation, use proper sparse matrix
                pass
            
            # Solve (simplified - in practice use sparse solver)
            # For now, just apply small corrections
            if len(self.loop_closures) > 0:
                # Apply loop closure corrections
                self.apply_loop_closure_corrections(optimized_poses)
        
        return optimized_poses
    
    def compute_residual(self, pose_i: np.ndarray, pose_j: np.ndarray, 
                        relative_pose: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute residual and Jacobian"""
        # Compute predicted relative pose
        dx = pose_j[0] - pose_i[0]
        dy = pose_j[1] - pose_i[1]
        dyaw = pose_j[2] - pose_i[2]
        
        # Normalize yaw
        dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
        
        predicted = np.array([dx, dy, dyaw])
        
        # Residual
        residual = predicted - relative_pose
        
        # Normalize yaw residual
        residual[2] = math.atan2(math.sin(residual[2]), math.cos(residual[2]))
        
        # Weight by covariance
        inv_cov = np.linalg.inv(covariance)
        weighted_residual = inv_cov @ residual
        
        # Simplified Jacobian (identity for now)
        jacobian = np.eye(3)
        
        return weighted_residual, jacobian
    
    def apply_loop_closure_corrections(self, poses: List[np.ndarray]):
        """Apply loop closure corrections (simplified)"""
        if not self.loop_closures:
            return
        
        # Simple correction: distribute error along path
        for edge in self.loop_closures:
            i, j = edge['from'], edge['to']
            if j >= len(poses):
                continue
            
            # Compute error
            dx = poses[j][0] - poses[i][0]
            dy = poses[j][1] - poses[i][1]
            dyaw = poses[j][2] - poses[i][2]
            
            expected = edge['relative_pose']
            error = np.array([dx, dy, dyaw]) - expected
            
            # Distribute correction
            num_nodes = j - i
            if num_nodes > 0:
                correction_per_node = error / num_nodes
                
                for k in range(i + 1, j + 1):
                    alpha = (k - i) / num_nodes
                    poses[k][0] -= error[0] * alpha * 0.3  # Damping
                    poses[k][1] -= error[1] * alpha * 0.3
                    poses[k][2] -= error[2] * alpha * 0.3


class GraphOptimizer:
    """Graph Optimizer wrapper"""
    
    def __init__(self):
        self.pose_graph = PoseGraph()
        self.node_indices = {}  # Map pose_id to graph node index
    
    def add_pose(self, pose_id: str, pose: np.ndarray):
        """Add pose to graph"""
        node_idx = self.pose_graph.add_node(pose)
        self.node_indices[pose_id] = node_idx
        return node_idx
    
    def add_odometry(self, pose_id_i: str, pose_id_j: str, 
                    relative_pose: np.ndarray, covariance: np.ndarray = None):
        """Add odometry constraint"""
        if pose_id_i not in self.node_indices or pose_id_j not in self.node_indices:
            return
        
        i = self.node_indices[pose_id_i]
        j = self.node_indices[pose_id_j]
        self.pose_graph.add_edge(i, j, relative_pose, covariance)
    
    def add_loop_closure(self, pose_id_i: str, pose_id_j: str,
                        relative_pose: np.ndarray, covariance: np.ndarray = None):
        """Add loop closure constraint"""
        if pose_id_i not in self.node_indices or pose_id_j not in self.node_indices:
            return
        
        i = self.node_indices[pose_id_i]
        j = self.node_indices[pose_id_j]
        self.pose_graph.add_loop_closure(i, j, relative_pose, covariance)
    
    def optimize(self, iterations: int = 10) -> Dict[str, np.ndarray]:
        """Optimize and return updated poses"""
        optimized_poses = self.pose_graph.optimize(iterations)
        
        # Map back to pose IDs
        result = {}
        for pose_id, node_idx in self.node_indices.items():
            if node_idx < len(optimized_poses):
                result[pose_id] = optimized_poses[node_idx]
        
        return result

