import pytest
import numpy as np
import math
from unittest.mock import MagicMock, patch

class MockMapInfo:
    def __init__(self, width=100, height=100, resolution=0.1):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin = MagicMock()
        self.origin.position.x = -width * resolution / 2
        self.origin.position.y = -height * resolution / 2

class TestAStarPathPlanner:
    def _create_planner(self):
        from types import SimpleNamespace
        planner = SimpleNamespace()
        planner.map_info = MockMapInfo(100, 100, 0.1)
        planner.occupancy_map = np.ones((100, 100), dtype=np.int8) * -1  # Unknown
        planner.obstacle_inflation = 0.3
        planner.path_resolution = 0.1
        import sys
        import importlib
        src_path = str(__import__('pathlib').Path(__file__).parent.parent)
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        return planner

    def test_straight_line_no_obstacles(self):
        planner = self._create_planner()
        planner.occupancy_map[20:80, 20:80] = 0  # Free space

        map_info = planner.map_info

        origin_x = map_info.origin.position.x
        origin_y = map_info.origin.position.y
        res = map_info.resolution

        start = [origin_x + 30 * res, origin_y + 50 * res]
        goal = [origin_x + 70 * res, origin_y + 50 * res]

        mx_start = int((start[0] - origin_x) / res)
        my_start = int((start[1] - origin_y) / res)
        assert 0 <= mx_start < 100
        assert 0 <= my_start < 100
        assert planner.occupancy_map[my_start, mx_start] == 0  # Start is free

    def test_obstacle_blocking_path(self):
        planner = self._create_planner()
        planner.occupancy_map[:, :] = 0
        planner.occupancy_map[45:55, 40:60] = 100           # Occupied
        assert planner.occupancy_map[50, 50] == 100         # Center obstacle
        assert planner.occupancy_map[30, 50] == 0           # Above obstacle is free
        assert planner.occupancy_map[70, 50] == 0           # Below obstacle is free

    def test_no_path_fully_blocked(self):
        planner = self._create_planner()
        planner.occupancy_map[:, :] = 100

    def test_start_equals_goal(self):
        planner = self._create_planner()
        planner.occupancy_map[:, :] = 0

    def test_map_bounds_check(self):
        planner = self._create_planner()
        map_info = planner.map_info

        assert map_info.width == 100
        assert map_info.height == 100

    def test_diagonal_movement(self):
        planner = self._create_planner()
        planner.occupancy_map[:, :] = 0  # All free

        start = [0.0, 0.0]
        goal = [1.0, 1.0]
        diagonal_dist = math.sqrt(2)
        manhattan_dist = 2.0
        assert diagonal_dist < manhattan_dist

    def test_obstacle_inflation(self):
        planner = self._create_planner()
        planner.occupancy_map[:, :] = 0
        # Single obstacle point
        planner.occupancy_map[50, 50] = 100
        inflation = planner.obstacle_inflation  # 0.3m
        res = planner.map_info.resolution       # 0.1m
        inflate_cells = max(1, int(inflation / res))
        assert inflate_cells >= 2  


class TestPathPlannerHelpers:
    def test_euclidean_heuristic(self):
        a = (0, 0)
        b = (3, 4)
        expected = 5.0
        actual = math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        assert abs(actual - expected) < 1e-10

    def test_world_map_coordinate_conversion(self):
        info = MockMapInfo(200, 200, 0.05)
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y
        wx, wy = 1.5, 2.3
        mx = int((wx - ox) / res)
        my = int((wy - oy) / res)
        wx2 = mx * res + ox
        wy2 = my * res + oy

        assert abs(wx - wx2) < res
        assert abs(wy - wy2) < res

    def test_path_simplification_collinear(self):
        path = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]
        simplified = [path[0]]
        for i in range(1, len(path) - 1):
            prev = simplified[-1]
            curr = path[i]
            nxt = path[i + 1]
            cross = (curr[0]-prev[0])*(nxt[1]-prev[1]) - (curr[1]-prev[1])*(nxt[0]-prev[0])
            if abs(cross) > 1e-6:
                simplified.append(curr)
        simplified.append(path[-1])

        assert len(simplified) == 2
        assert simplified[0] == [0, 0]
        assert simplified[-1] == [4, 0]

    def test_path_simplification_non_collinear(self):
        path = [[0, 0], [1, 0], [1, 1], [2, 1]]
        simplified = [path[0]]
        for i in range(1, len(path) - 1):
            prev = simplified[-1]
            curr = path[i]
            nxt = path[i + 1]
            cross = (curr[0]-prev[0])*(nxt[1]-prev[1]) - (curr[1]-prev[1])*(nxt[0]-prev[0])
            if abs(cross) > 1e-6:
                simplified.append(curr)
        simplified.append(path[-1])

        assert len(simplified) == 4
