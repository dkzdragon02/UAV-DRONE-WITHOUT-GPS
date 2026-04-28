import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from uav_vision.obstacle_avoidance import ObstacleAvoidance

class TestObstacleAvoidance:
    def setup_method(self):
        self.oa = ObstacleAvoidance(repulsive_gain=1.0, attractive_gain=1.0)
        self.oa.safety_distance = 0.5
        self.oa.max_repulsive_distance = 2.0
        self.oa.emergency_distance = 0.3

    def test_attractive_force_direction(self):
        pos = np.array([0.0, 0.0])
        goal = np.array([5.0, 0.0])
        force = self.oa.compute_attractive_force(pos, goal)
        assert force[0] > 0, "Should pull towards positive x"
        assert abs(force[1]) < 1e-9, "No y component expected"

    def test_attractive_force_zero_at_goal(self):
        pos = np.array([3.0, 4.0])
        goal = np.array([3.0, 4.0])
        force = self.oa.compute_attractive_force(pos, goal)
        assert np.linalg.norm(force) < 1e-6

    def test_attractive_force_magnitude_bounded(self):
        pos = np.array([0.0, 0.0])
        goal = np.array([100.0, 0.0])
        force = self.oa.compute_attractive_force(pos, goal)
        assert np.linalg.norm(force) <= 1.0 + 1e-9

    def test_repulsive_force_pushes_away(self):
        pos = np.array([0.0, 0.0])
        obs = np.array([1.0, 0.0])
        force = self.oa.compute_repulsive_force(pos, obs)
        assert force[0] < 0, "Should push away from obstacle (negative x)"

    def test_repulsive_force_zero_far_away(self):
        pos = np.array([0.0, 0.0])
        obs = np.array([10.0, 0.0])  # Far beyond max_repulsive_distance=2.0
        force = self.oa.compute_repulsive_force(pos, obs)
        assert np.linalg.norm(force) < 1e-9

    def test_repulsive_force_strong_when_close(self):
        pos = np.array([0.0, 0.0])
        close_obs = np.array([0.3, 0.0])
        far_obs = np.array([1.5, 0.0])
        close_force = np.linalg.norm(self.oa.compute_repulsive_force(pos, close_obs))
        far_force = np.linalg.norm(self.oa.compute_repulsive_force(pos, far_obs))
        assert close_force > far_force * 5, "Close obstacle force should be much larger"

    def test_repulsive_force_at_zero_distance(self):
        pos = np.array([0.0, 0.0])
        obs = np.array([0.0, 0.0])
        force = self.oa.compute_repulsive_force(pos, obs)
        assert np.linalg.norm(force) > 0, "Should produce escape force"

    def test_potential_field_no_obstacles(self):
        pos = np.array([0.0, 0.0])
        goal = np.array([5.0, 0.0])
        force = self.oa.compute_potential_field(pos, goal, [])
        attractive = self.oa.compute_attractive_force(pos, goal)
        np.testing.assert_allclose(force, attractive)

    def test_potential_field_balanced(self):
        pos = np.array([0.0, 0.0])
        goal = np.array([5.0, 0.0])
        obstacles = [np.array([0.0, 1.0]), np.array([0.0, -1.0])]
        force = self.oa.compute_potential_field(pos, goal, obstacles)
        assert abs(force[1]) < 0.1, "Symmetric obstacles should mostly cancel y-force"

    def test_3d_potential_field(self):
        pos = np.array([0.0, 0.0, 2.0])
        goal = np.array([5.0, 0.0, 2.0])
        obs = np.array([1.0, 0.0, 2.5])  # Obstacle slightly above
        force = self.oa.compute_potential_field_3d(pos, goal, [obs])
        assert len(force) == 3
        assert force[2] < 0, "Should push away from obstacle above (negative z)"

    def test_3d_with_2d_obstacles(self):
        pos = np.array([0.0, 0.0, 2.0])
        goal = np.array([5.0, 0.0, 2.0])
        obs_2d = np.array([1.0, 0.0])
        force = self.oa.compute_potential_field_3d(pos, goal, [obs_2d])
        assert len(force) == 3

    def test_velocity_correction_with_obstacle(self):
        pos = np.array([0.0, 0.0])
        obs = [np.array([0.5, 0.0])]
        correction = self.oa.compute_velocity_correction(pos, obs, max_correction=1.0)
        assert correction[0] < 0, "Should correct away from obstacle"

    def test_velocity_correction_no_obstacles(self):
        pos = np.array([0.0, 0.0])
        correction = self.oa.compute_velocity_correction(pos, [], max_correction=1.0)
        assert np.linalg.norm(correction) < 1e-9

    def test_velocity_correction_clamped(self):
        pos = np.array([0.0, 0.0])
        obs = [np.array([0.1, 0.0])]  # Very close
        correction = self.oa.compute_velocity_correction(pos, obs, max_correction=0.5)
        assert np.linalg.norm(correction) <= 0.5 + 1e-6

    def test_nearest_obstacle_dist_empty(self):
        pos = np.array([0.0, 0.0])
        dist = self.oa.get_nearest_obstacle_distance(pos, [])
        assert dist == float('inf')

    def test_nearest_obstacle_dist_correct(self):
        pos = np.array([0.0, 0.0])
        obstacles = [np.array([3.0, 0.0]), np.array([1.0, 0.0]), np.array([5.0, 0.0])]
        dist = self.oa.get_nearest_obstacle_distance(pos, obstacles)
        assert abs(dist - 1.0) < 1e-6

    def test_emergency_stop_triggered(self):
        pos = np.array([0.0, 0.0])
        obs = [np.array([0.2, 0.0])]  # < 0.3m emergency_distance
        assert self.oa.is_emergency_stop(pos, obs) is True

    def test_emergency_stop_not_triggered(self):
        pos = np.array([0.0, 0.0])
        obs = [np.array([1.0, 0.0])]
        assert self.oa.is_emergency_stop(pos, obs) is False

    def test_collision_detected(self):
        pos = np.array([0.0, 0.0])
        obs = [np.array([0.1, 0.0])]
        assert self.oa.check_collision(pos, obs, obstacle_radius=0.3) is True

    def test_no_collision(self):
        pos = np.array([0.0, 0.0])
        obs = [np.array([5.0, 0.0])]
        assert self.oa.check_collision(pos, obs, obstacle_radius=0.3) is False

    def test_obstacle_history_update(self):
        obs1 = [np.array([1.0, 0.0])]
        obs2 = [np.array([0.9, 0.0])]
        self.oa.update_obstacle_history(obs1)
        self.oa.update_obstacle_history(obs2)
        assert len(self.oa._obstacle_history) == 2

    def test_predict_obstacles(self):
        self.oa._obstacle_history = []
        import time
        obs1 = [np.array([1.0, 0.0])]
        self.oa._obstacle_history.append((time.time() - 0.1, obs1))
        obs2 = [np.array([0.9, 0.0])]
        self.oa._obstacle_history.append((time.time(), obs2))
        predicted = self.oa.predict_obstacles(dt=0.1)
        assert len(predicted) == 1
        # Moving from 1.0 to 0.9 in 0.1s → velocity = -1.0/s → at dt=0.1: 0.9 + (-1.0)*0.1 = 0.8
        assert predicted[0][0] < 0.9, "Should predict obstacle moving closer"

    def test_find_safe_path_no_obstacles(self):
        start = np.array([0.0, 0.0])
        goal = np.array([2.0, 0.0])
        path = self.oa.find_safe_path(start, goal, [], num_waypoints=20)
        assert len(path) > 0
        # Last waypoint should be near goal
        assert np.linalg.norm(path[-1] - goal) < 1.0

    def test_avoid_obstacles_velocity_clamped(self):
        pos = np.array([0.0, 0.0])
        goal = np.array([10.0, 0.0])
        vel = self.oa.avoid_obstacles(pos, goal, [], max_velocity=0.5)
        assert np.linalg.norm(vel) <= 0.5 + 1e-6
