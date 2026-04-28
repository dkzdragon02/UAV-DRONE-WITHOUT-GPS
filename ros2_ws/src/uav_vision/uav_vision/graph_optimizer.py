import numpy as np
from typing import List, Tuple, Dict
import math

class PoseGraph:
    def __init__(self):
        self.nodes = []             # List of poses [x, y, yaw]
        self.edges = []             # List of constraints [(i, j, relative_pose, covariance)]
        self.loop_closures = []     # List of loop closure edges
    
    def add_node(self, pose: np.ndarray):
        self.nodes.append(pose.copy())
        return len(self.nodes) - 1
    
    def add_edge(self, i: int, j: int, relative_pose: np.ndarray, covariance: np.ndarray = None):
        if covariance is None:
            covariance = np.eye(3) * 0.1
        
        self.edges.append({
            'from': i,
            'to': j,
            'relative_pose': relative_pose.copy(),
            'covariance': covariance.copy()
        })
    
    def add_loop_closure(self, i: int, j: int, relative_pose: np.ndarray, covariance: np.ndarray = None):
        if covariance is None:
            covariance = np.eye(3) * 0.05  # Loop closures are more certain
        
        self.loop_closures.append({
            'from': i,
            'to': j,
            'relative_pose': relative_pose.copy(),
            'covariance': covariance.copy()
        })
    
    def optimize(self, iterations: int = 10, convergence_tol: float = 1e-6) -> List[np.ndarray]:
        n = len(self.nodes)
        if n < 2:
            return self.nodes.copy()

        optimized_poses = [pose.copy() for pose in self.nodes]
        dim = 3  
        total_dim = n * dim
        all_edges = self.edges + self.loop_closures

        for iteration in range(iterations):
            H = np.zeros((total_dim, total_dim))
            b = np.zeros(total_dim)

            for edge in all_edges:
                i, j = edge['from'], edge['to']
                if i >= n or j >= n:
                    continue

                pi = optimized_poses[i]
                pj = optimized_poses[j]
                z_ij = edge['relative_pose']
                omega = np.linalg.inv(edge['covariance'])
                cos_i = math.cos(pi[2])
                sin_i = math.sin(pi[2])
                dx = pj[0] - pi[0]
                dy = pj[1] - pi[1]
                predicted = np.array([
                    cos_i * dx + sin_i * dy,
                    -sin_i * dx + cos_i * dy,
                    math.atan2(math.sin(pj[2] - pi[2]),
                               math.cos(pj[2] - pi[2])),
                ])

                e = predicted - z_ij
                e[2] = math.atan2(math.sin(e[2]), math.cos(e[2]))

                Ji = np.array([
                    [-cos_i, -sin_i,  -sin_i * dx + cos_i * dy],
                    [ sin_i, -cos_i,  -cos_i * dx - sin_i * dy],
                    [  0.0,    0.0,   -1.0],
                ])
                Jj = np.array([
                    [ cos_i,  sin_i,  0.0],
                    [-sin_i,  cos_i,  0.0],
                    [  0.0,    0.0,   1.0],
                ])

                ii = i * dim
                jj = j * dim

                H[ii:ii+dim, ii:ii+dim] += Ji.T @ omega @ Ji
                H[ii:ii+dim, jj:jj+dim] += Ji.T @ omega @ Jj
                H[jj:jj+dim, ii:ii+dim] += Jj.T @ omega @ Ji
                H[jj:jj+dim, jj:jj+dim] += Jj.T @ omega @ Jj
                b[ii:ii+dim] += Ji.T @ omega @ e
                b[jj:jj+dim] += Jj.T @ omega @ e
            H[0:dim, 0:dim] += np.eye(dim) * 1e6
            damping = 1e-3
            try:
                dx = np.linalg.solve(H + damping * np.eye(total_dim), -b)
            except np.linalg.LinAlgError:
                if self.loop_closures:
                    self.apply_loop_closure_corrections(optimized_poses)        # Singular — fall back to legacy correction
                break

            for k in range(n):
                ki = k * dim
                optimized_poses[k][0] += dx[ki]
                optimized_poses[k][1] += dx[ki + 1]
                optimized_poses[k][2] += dx[ki + 2]
                optimized_poses[k][2] = math.atan2(
                    math.sin(optimized_poses[k][2]),
                    math.cos(optimized_poses[k][2]),
                )

            if np.linalg.norm(dx) < convergence_tol:
                break

        return optimized_poses
    
    def compute_residual(self, pose_i: np.ndarray, pose_j: np.ndarray, 
                        relative_pose: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dx = pose_j[0] - pose_i[0]
        dy = pose_j[1] - pose_i[1]
        dyaw = pose_j[2] - pose_i[2]
        dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
        predicted = np.array([dx, dy, dyaw])
        residual = predicted - relative_pose
        residual[2] = math.atan2(math.sin(residual[2]), math.cos(residual[2]))
        inv_cov = np.linalg.inv(covariance)
        weighted_residual = inv_cov @ residual
        jacobian = np.eye(3)
        
        return weighted_residual, jacobian
    
    def apply_loop_closure_corrections(self, poses: List[np.ndarray]):
        if not self.loop_closures:
            return
        
        for edge in self.loop_closures:
            i, j = edge['from'], edge['to']
            if j >= len(poses):
                continue
            
            dx = poses[j][0] - poses[i][0]
            dy = poses[j][1] - poses[i][1]
            dyaw = poses[j][2] - poses[i][2]
            
            expected = edge['relative_pose']
            error = np.array([dx, dy, dyaw]) - expected
            
            num_nodes = j - i
            if num_nodes > 0:
                correction_per_node = error / num_nodes
                
                for k in range(i + 1, j + 1):
                    alpha = (k - i) / num_nodes
                    poses[k][0] -= error[0] * alpha * 0.3   # Damping
                    poses[k][1] -= error[1] * alpha * 0.3
                    poses[k][2] -= error[2] * alpha * 0.3

class GraphOptimizer:
    def __init__(self):
        self.pose_graph = PoseGraph()
        self.node_indices = {}                              # Map pose_id to graph node index
    
    def add_pose(self, pose_id: str, pose: np.ndarray):
        node_idx = self.pose_graph.add_node(pose)
        self.node_indices[pose_id] = node_idx
        return node_idx
    
    def add_odometry(self, pose_id_i: str, pose_id_j: str, 
                    relative_pose: np.ndarray, covariance: np.ndarray = None):
        if pose_id_i not in self.node_indices or pose_id_j not in self.node_indices:
            return
        
        i = self.node_indices[pose_id_i]
        j = self.node_indices[pose_id_j]
        self.pose_graph.add_edge(i, j, relative_pose, covariance)
    
    def add_loop_closure(self, pose_id_i: str, pose_id_j: str,
                        relative_pose: np.ndarray, covariance: np.ndarray = None):
        if pose_id_i not in self.node_indices or pose_id_j not in self.node_indices:
            return
        
        i = self.node_indices[pose_id_i]
        j = self.node_indices[pose_id_j]
        self.pose_graph.add_loop_closure(i, j, relative_pose, covariance)
    
    def optimize(self, iterations: int = 10) -> Dict[str, np.ndarray]:
        optimized_poses = self.pose_graph.optimize(iterations)
        result = {}
        for pose_id, node_idx in self.node_indices.items():
            if node_idx < len(optimized_poses):
                result[pose_id] = optimized_poses[node_idx]
        
        return result

