import numpy as np
from typing import List, Tuple, Dict
from collections import deque
import time
import math

class SLAMEvaluator:  
    def __init__(self):
        self.ground_truth_poses = []                # Ground truth poses (if available)
        self.estimated_poses = []
        self.timestamps = []
        self.errors = {
            'position': [],
            'orientation': [],
            'trajectory': []
        }
        
        self.stats = {
            'total_frames': 0,
            'successful_frames': 0,
            'loop_closures': 0,
            'drift': 0.0,
            'processing_time': deque(maxlen=100)
        }
    
    def add_pose(self, estimated_pose: np.ndarray, ground_truth: np.ndarray = None, timestamp: float = None):
        self.estimated_poses.append(estimated_pose.copy())
        
        if ground_truth is not None:
            self.ground_truth_poses.append(ground_truth.copy())
        
        if timestamp is None:
            timestamp = time.time()
        self.timestamps.append(timestamp)
        
        if ground_truth is not None and len(self.ground_truth_poses) > 0:
            error = self.compute_error(estimated_pose, ground_truth)
            self.errors['position'].append(error['position'])
            self.errors['orientation'].append(error['orientation'])
    
    def compute_error(self, estimated: np.ndarray, ground_truth: np.ndarray) -> Dict:
        pos_error = np.linalg.norm(estimated[:2] - ground_truth[:2])
        yaw_error = abs(estimated[2] - ground_truth[2])
        yaw_error = min(yaw_error, 2 * math.pi - yaw_error)  # Wrap to [0, pi]
        
        return {
            'position': pos_error,
            'orientation': yaw_error
        }
    
    def compute_metrics(self) -> Dict:
        metrics = {}
        
        if len(self.errors['position']) > 0:
            metrics['mae_position'] = np.mean(self.errors['position'])
            metrics['rmse_position'] = np.sqrt(np.mean([e**2 for e in self.errors['position']]))
            metrics['max_position_error'] = np.max(self.errors['position'])
            metrics['std_position_error'] = np.std(self.errors['position'])
        
        if len(self.errors['orientation']) > 0:
            metrics['mae_orientation'] = np.mean(self.errors['orientation'])
            metrics['rmse_orientation'] = np.sqrt(np.mean([e**2 for e in self.errors['orientation']]))
            metrics['max_orientation_error'] = np.max(self.errors['orientation'])
        
        if len(self.estimated_poses) > 1:
            metrics['total_distance'] = self.compute_trajectory_length()
            metrics['drift_rate'] = self.compute_drift_rate()
        
        if len(self.stats['processing_time']) > 0:
            metrics['avg_processing_time'] = np.mean(self.stats['processing_time'])
            metrics['max_processing_time'] = np.max(self.stats['processing_time'])
            metrics['min_processing_time'] = np.min(self.stats['processing_time'])
        
        if self.stats['total_frames'] > 0:
            metrics['success_rate'] = self.stats['successful_frames'] / self.stats['total_frames']
        
        return metrics
    
    def compute_trajectory_length(self) -> float:
        if len(self.estimated_poses) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(self.estimated_poses)):
            dx = self.estimated_poses[i][0] - self.estimated_poses[i-1][0]
            dy = self.estimated_poses[i][1] - self.estimated_poses[i-1][1]
            total_length += math.sqrt(dx*dx + dy*dy)
        
        return total_length
    
    def compute_drift_rate(self) -> float:
        if len(self.estimated_poses) < 2 or len(self.timestamps) < 2:
            return 0.0

        start_pos = self.estimated_poses[0][:2]
        end_pos = self.estimated_poses[-1][:2]
        drift = np.linalg.norm(end_pos - start_pos)
        time_elapsed = self.timestamps[-1] - self.timestamps[0]
        
        if time_elapsed > 0:
            return drift / time_elapsed
        
        return 0.0
    
    def record_processing_time(self, time_ms: float):
        self.stats['processing_time'].append(time_ms)
    
    def record_success(self, success: bool):
        self.stats['total_frames'] += 1
        if success:
            self.stats['successful_frames'] += 1
    
    def record_loop_closure(self):
        self.stats['loop_closures'] += 1
    
    def get_report(self) -> str:
        metrics = self.compute_metrics()
        
        report = "=== SLAM Evaluation Report ===\n\n"
        report += f"Total Frames: {self.stats['total_frames']}\n"
        report += f"Successful Frames: {self.stats['successful_frames']}\n"
        report += f"Success Rate: {metrics.get('success_rate', 0.0)*100:.2f}%\n"
        report += f"Loop Closures: {self.stats['loop_closures']}\n\n"
        
        if 'mae_position' in metrics:
            report += "Position Errors:\n"
            report += f"  MAE: {metrics['mae_position']:.3f} m\n"
            report += f"  RMSE: {metrics['rmse_position']:.3f} m\n"
            report += f"  Max: {metrics['max_position_error']:.3f} m\n"
            report += f"  Std: {metrics['std_position_error']:.3f} m\n\n"
        
        if 'mae_orientation' in metrics:
            report += "Orientation Errors:\n"
            report += f"  MAE: {metrics['mae_orientation']:.3f} rad\n"
            report += f"  RMSE: {metrics['rmse_orientation']:.3f} rad\n"
            report += f"  Max: {metrics['max_orientation_error']:.3f} rad\n\n"
        
        if 'total_distance' in metrics:
            report += f"Trajectory Length: {metrics['total_distance']:.2f} m\n"
            report += f"Drift Rate: {metrics.get('drift_rate', 0.0):.4f} m/s\n\n"
        
        if 'avg_processing_time' in metrics:
            report += "Performance:\n"
            report += f"  Avg Processing Time: {metrics['avg_processing_time']:.2f} ms\n"
            report += f"  Max Processing Time: {metrics['max_processing_time']:.2f} ms\n"
            report += f"  Min Processing Time: {metrics['min_processing_time']:.2f} ms\n"
        
        return report


class NavigationEvaluator:  
    def __init__(self):
        self.waypoint_errors = []
        self.path_following_errors = []
        self.mission_times = []
        
    def record_waypoint_error(self, error: float):
        self.waypoint_errors.append(error)
    
    def record_path_following_error(self, error: float):
        self.path_following_errors.append(error)
    
    def compute_metrics(self) -> Dict:
        metrics = {}
        
        if self.waypoint_errors:
            metrics['waypoint_mae'] = np.mean(self.waypoint_errors)
            metrics['waypoint_rmse'] = np.sqrt(np.mean([e**2 for e in self.waypoint_errors]))
            metrics['waypoint_max_error'] = np.max(self.waypoint_errors)
        
        if self.path_following_errors:
            metrics['path_following_mae'] = np.mean(self.path_following_errors)
            metrics['path_following_rmse'] = np.sqrt(np.mean([e**2 for e in self.path_following_errors]))
        
        return metrics

