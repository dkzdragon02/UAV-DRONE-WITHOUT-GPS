import numpy as np
from typing import List, Tuple, Optional
import math
import time

class ObstacleAvoidance:
    def __init__(self, repulsive_gain: float = 1.0, attractive_gain: float = 1.0):
        self.repulsive_gain = repulsive_gain
        self.attractive_gain = attractive_gain
        self.safety_distance = 0.5              # meters
        self.max_repulsive_distance = 2.0       # meters
        self.emergency_distance = 0.3           # meters — trigger emergency stop
        self._obstacle_history: List[Tuple[float, List[np.ndarray]]] = []
        self._history_max_len = 5               # keep last N frames
        self._history_max_age = 2.0             # seconds

    def compute_potential_field(self, current_pos: np.ndarray, goal_pos: np.ndarray,
                               obstacles: List[np.ndarray]) -> np.ndarray:

        attractive_force = self.compute_attractive_force(current_pos, goal_pos)
        repulsive_force = np.zeros_like(current_pos, dtype=float)
        for obstacle in obstacles:
            repulsive = self.compute_repulsive_force(current_pos, obstacle)
            repulsive_force += repulsive
        total_force = self.attractive_gain * attractive_force + self.repulsive_gain * repulsive_force
        return total_force

    def compute_attractive_force(self, current_pos: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
        direction = goal_pos - current_pos
        distance = np.linalg.norm(direction)

        if distance < 0.01:
            return np.zeros_like(current_pos, dtype=float)

        direction = direction / distance
        magnitude = min(distance, 1.0)  

        return direction * magnitude

    def compute_repulsive_force(self, current_pos: np.ndarray, obstacle_pos: np.ndarray) -> np.ndarray:
        direction = current_pos - obstacle_pos
        distance = np.linalg.norm(direction)

        if distance > self.max_repulsive_distance:
            return np.zeros_like(current_pos, dtype=float)

        if distance < 0.01:
            result = np.zeros_like(current_pos, dtype=float)
            result[0] = 10.0
            return result
        direction = direction / distance

        if distance < self.safety_distance:
            magnitude = 1.0 / (distance ** 2)
        else:
            magnitude = 1.0 / (distance ** 2) * (self.max_repulsive_distance - distance) / self.max_repulsive_distance

        return direction * magnitude

    def avoid_obstacles(self, current_pos: np.ndarray, goal_pos: np.ndarray,
                       obstacles: List[np.ndarray], max_velocity: float = 1.0) -> np.ndarray:
        force = self.compute_potential_field(current_pos, goal_pos, obstacles)
        velocity = force * max_velocity
        speed = np.linalg.norm(velocity)
        if speed > max_velocity:
            velocity = velocity / speed * max_velocity

        return velocity

    def check_collision(self, pos: np.ndarray, obstacles: List[np.ndarray],
                       obstacle_radius: float = 0.3) -> bool:
        for obstacle in obstacles:
            distance = np.linalg.norm(pos[:len(obstacle)] - obstacle[:len(pos)])
            if distance < obstacle_radius:
                return True
        return False

    def find_safe_path(self, start: np.ndarray, goal: np.ndarray,
                      obstacles: List[np.ndarray], num_waypoints: int = 10) -> List[np.ndarray]:
        waypoints = []
        current = start.copy()
        step_size = 0.2  # meters

        for i in range(num_waypoints):
            distance_to_goal = np.linalg.norm(current - goal)
            if distance_to_goal < 0.5:
                waypoints.append(goal.copy())
                break

            force = self.compute_potential_field(current, goal, obstacles)
            force_norm = np.linalg.norm(force)
            if force_norm > 0.01:
                force = force / force_norm * step_size
            next_pos = current + force

            if not self.check_collision(next_pos, obstacles):
                current = next_pos
                waypoints.append(current.copy())
            else:
                if obstacles:
                    closest_obstacle = min(obstacles, key=lambda o: np.linalg.norm(current - o))
                    direction = current - closest_obstacle
                    direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0.01 else np.array([1.0, 0.0])
                    perp = np.array([-direction[1], direction[0]])
                    next_pos = current + perp * step_size

                    if not self.check_collision(next_pos, obstacles):
                        current = next_pos
                        waypoints.append(current.copy())

        return waypoints

    def compute_potential_field_3d(self, current_pos: np.ndarray, goal_pos: np.ndarray,
                                  obstacles: List[np.ndarray]) -> np.ndarray:
        attractive_force = self.compute_attractive_force(current_pos, goal_pos)
        repulsive_force = np.zeros(3, dtype=float)
        for obstacle in obstacles:
            obs = obstacle if len(obstacle) == 3 else np.array([obstacle[0], obstacle[1], current_pos[2]])
            repulsive = self.compute_repulsive_force(current_pos, obs)
            repulsive_force += repulsive

        total_force = self.attractive_gain * attractive_force + self.repulsive_gain * repulsive_force
        return total_force

    def compute_velocity_correction(self, current_pos: np.ndarray,
                                    obstacles: List[np.ndarray],
                                    max_correction: float = 1.0) -> np.ndarray:
        dim = len(current_pos)
        correction = np.zeros(dim, dtype=float)

        for obstacle in obstacles:
            obs = np.array(obstacle[:dim], dtype=float) if len(obstacle) >= dim else np.zeros(dim)
            obs[:len(obstacle)] = obstacle[:min(len(obstacle), dim)]
            rep = self.compute_repulsive_force(current_pos, obs)
            correction += rep

        correction *= self.repulsive_gain
        speed = np.linalg.norm(correction)

        if speed > max_correction:
            correction = correction / speed * max_correction

        return correction

    def get_nearest_obstacle_distance(self, current_pos: np.ndarray,
                                      obstacles: List[np.ndarray]) -> float:
        if not obstacles:
            return float('inf')

        dim = len(current_pos)
        min_dist = float('inf')
        for obs in obstacles:
            obs_trimmed = np.array(obs[:dim], dtype=float)
            dist = float(np.linalg.norm(current_pos[:len(obs_trimmed)] - obs_trimmed))
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def is_emergency_stop(self, current_pos: np.ndarray,
                          obstacles: List[np.ndarray]) -> bool:
        return self.get_nearest_obstacle_distance(current_pos, obstacles) < self.emergency_distance

    def update_obstacle_history(self, obstacles: List[np.ndarray]) -> None:
        now = time.time()
        self._obstacle_history.append((now, [o.copy() for o in obstacles]))
        cutoff = now - self._history_max_age
        self._obstacle_history = [
            (t, obs) for t, obs in self._obstacle_history if t >= cutoff
        ]
        if len(self._obstacle_history) > self._history_max_len:
            self._obstacle_history = self._obstacle_history[-self._history_max_len:]

    def predict_obstacles(self, dt: float = 0.5) -> List[np.ndarray]:
        if len(self._obstacle_history) < 2:
            if self._obstacle_history:
                return self._obstacle_history[-1][1]
            return []

        t0, obs0 = self._obstacle_history[-2]
        t1, obs1 = self._obstacle_history[-1]
        elapsed = t1 - t0
        if elapsed < 0.01:
            return obs1

        predicted = []
        min_count = min(len(obs0), len(obs1))
        for i in range(min_count):
            velocity = (obs1[i] - obs0[i]) / elapsed
            predicted.append(obs1[i] + velocity * dt)
        for i in range(min_count, len(obs1)):
            predicted.append(obs1[i].copy())

        return predicted
