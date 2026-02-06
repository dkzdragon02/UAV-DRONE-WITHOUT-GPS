#!/usr/bin/env python3
"""
Testing Framework cho UAV Vision System
Unit tests và integration tests
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uav_vision.visual_odometry import VisualOdometry
from uav_vision.graph_optimizer import GraphOptimizer
from uav_vision.coverage_planner import CoveragePlanner
from uav_vision.obstacle_avoidance import ObstacleAvoidance
from uav_vision.evaluation import SLAMEvaluator


class TestVisualOdometry(unittest.TestCase):
    """Test Visual Odometry"""
    
    def setUp(self):
        self.vo = VisualOdometry(feature_detector="ORB", max_features=100)
    
    def test_initialization(self):
        """Test VO initialization"""
        self.assertIsNotNone(self.vo.detector)
        self.assertIsNotNone(self.vo.matcher)
        np.testing.assert_array_equal(self.vo.position, np.array([0.0, 0.0, 0.0]))
    
    def test_camera_params(self):
        """Test camera parameters setting"""
        camera_matrix = np.eye(3) * 500
        dist_coeffs = np.zeros(5)
        self.vo.set_camera_params(camera_matrix, dist_coeffs)
        np.testing.assert_array_equal(self.vo.camera_matrix, camera_matrix)
        np.testing.assert_array_equal(self.vo.dist_coeffs, dist_coeffs)


class TestGraphOptimizer(unittest.TestCase):
    """Test Graph Optimizer"""
    
    def setUp(self):
        self.optimizer = GraphOptimizer()
    
    def test_add_pose(self):
        """Test adding poses"""
        pose1 = np.array([0.0, 0.0, 0.0])
        pose2 = np.array([1.0, 0.0, 0.0])
        
        idx1 = self.optimizer.add_pose("pose1", pose1)
        idx2 = self.optimizer.add_pose("pose2", pose2)
        
        self.assertEqual(len(self.optimizer.pose_graph.nodes), 2)
        self.assertIn("pose1", self.optimizer.node_indices)
        self.assertIn("pose2", self.optimizer.node_indices)
    
    def test_add_odometry(self):
        """Test adding odometry constraint"""
        pose1 = np.array([0.0, 0.0, 0.0])
        pose2 = np.array([1.0, 0.0, 0.0])
        
        self.optimizer.add_pose("pose1", pose1)
        self.optimizer.add_pose("pose2", pose2)
        
        relative_pose = np.array([1.0, 0.0, 0.0])
        self.optimizer.add_odometry("pose1", "pose2", relative_pose)
        
        self.assertEqual(len(self.optimizer.pose_graph.edges), 1)


class TestCoveragePlanner(unittest.TestCase):
    """Test Coverage Planner"""
    
    def setUp(self):
        self.planner = CoveragePlanner()
    
    def test_lawnmower(self):
        """Test lawnmower pattern"""
        bounds = (0.0, 0.0, 10.0, 10.0)
        altitude = 2.0
        spacing = 1.0
        
        waypoints = self.planner.plan_lawnmower(bounds, altitude, spacing)
        
        self.assertGreater(len(waypoints), 0)
        self.assertEqual(waypoints[0][2], altitude)  # Check altitude
    
    def test_spiral(self):
        """Test spiral pattern"""
        center = (0.0, 0.0)
        max_radius = 5.0
        altitude = 2.0
        spacing = 0.5
        
        waypoints = self.planner.plan_spiral(center, max_radius, altitude, spacing)
        
        self.assertGreater(len(waypoints), 0)
        self.assertEqual(waypoints[0][2], altitude)


class TestObstacleAvoidance(unittest.TestCase):
    """Test Obstacle Avoidance"""
    
    def setUp(self):
        self.avoidance = ObstacleAvoidance()
    
    def test_attractive_force(self):
        """Test attractive force computation"""
        current = np.array([0.0, 0.0])
        goal = np.array([1.0, 0.0])
        
        force = self.avoidance.compute_attractive_force(current, goal)
        
        self.assertGreater(np.linalg.norm(force), 0)
        self.assertAlmostEqual(force[0], 1.0, places=1)
    
    def test_repulsive_force(self):
        """Test repulsive force computation"""
        current = np.array([0.0, 0.0])
        obstacle = np.array([0.5, 0.0])
        
        force = self.avoidance.compute_repulsive_force(current, obstacle)
        
        self.assertGreater(np.linalg.norm(force), 0)
        # Should push away from obstacle
        self.assertLess(force[0], 0)  # Negative x direction


class TestSLAMEvaluator(unittest.TestCase):
    """Test SLAM Evaluator"""
    
    def setUp(self):
        self.evaluator = SLAMEvaluator()
    
    def test_add_pose(self):
        """Test adding poses"""
        pose = np.array([1.0, 2.0, 0.5])
        self.evaluator.add_pose(pose)
        
        self.assertEqual(len(self.evaluator.estimated_poses), 1)
    
    def test_compute_metrics(self):
        """Test metrics computation"""
        # Add some poses
        for i in range(10):
            pose = np.array([i * 0.1, 0.0, 0.0])
            self.evaluator.add_pose(pose)
        
        metrics = self.evaluator.compute_metrics()
        
        self.assertIn('total_distance', metrics)
        self.assertGreater(metrics['total_distance'], 0)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestVisualOdometry))
    suite.addTests(loader.loadTestsFromTestCase(TestGraphOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestCoveragePlanner))
    suite.addTests(loader.loadTestsFromTestCase(TestObstacleAvoidance))
    suite.addTests(loader.loadTestsFromTestCase(TestSLAMEvaluator))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

