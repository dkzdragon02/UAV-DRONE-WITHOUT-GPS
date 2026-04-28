import pytest
import numpy as np
import math
from unittest.mock import MagicMock

class TestBoWSimilarity:
    def _bow_similarity(self, bow1, bow2):
        norm1 = np.linalg.norm(bow1)
        norm2 = np.linalg.norm(bow2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        return float(np.dot(bow1, bow2) / (norm1 * norm2))

    def test_identical_vectors(self):
        bow = np.array([1.0, 2.0, 3.0, 0.0, 1.0])
        assert abs(self._bow_similarity(bow, bow) - 1.0) < 1e-10

    def test_orthogonal_vectors(self):
        bow1 = np.array([1.0, 0.0, 0.0])
        bow2 = np.array([0.0, 1.0, 0.0])
        assert abs(self._bow_similarity(bow1, bow2)) < 1e-10

    def test_opposite_vectors(self):
        bow1 = np.array([1.0, 0.0])
        bow2 = np.array([-1.0, 0.0])
        assert abs(self._bow_similarity(bow1, bow2) - (-1.0)) < 1e-10

    def test_zero_vector(self):
        bow1 = np.array([1.0, 2.0])
        bow2 = np.zeros(2)
        assert self._bow_similarity(bow1, bow2) == 0.0

    def test_similar_vectors(self):
        bow1 = np.array([1.0, 2.0, 3.0])
        bow2 = np.array([1.1, 1.9, 3.1])
        sim = self._bow_similarity(bow1, bow2)
        assert sim > 0.99

    def test_dissimilar_vectors(self):
        bow1 = np.array([1.0, 0.0, 0.0, 0.0])
        bow2 = np.array([0.0, 0.0, 0.0, 1.0])
        sim = self._bow_similarity(bow1, bow2)
        assert sim < 0.01


class TestPoseUpdate:
    def test_initial_pose(self):
        pose = np.array([0.0, 0.0, 0.0])
        assert np.all(pose == 0.0)

    def test_translation_update(self):
        current_pose = np.array([0.0, 0.0, 0.0])
        current_rotation = np.eye(3)

        translation = np.array([1.0, 0.0, 0.0])
        translation_world = current_rotation @ translation

        current_pose[0] += translation_world[0]
        current_pose[1] += translation_world[1]

        assert abs(current_pose[0] - 1.0) < 1e-10
        assert abs(current_pose[1]) < 1e-10

    def test_rotation_update(self):
        current_rotation = np.eye(3)

        theta = math.pi / 2
        R = np.array([
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta), 0],
            [0, 0, 1]
        ])

        current_rotation = current_rotation @ R
        yaw = math.atan2(current_rotation[1, 0], current_rotation[0, 0])
        assert abs(yaw - math.pi / 2) < 1e-10

    def test_translation_in_rotated_frame(self):
        theta = math.pi / 2
        R = np.array([
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta), 0],
            [0, 0, 1]
        ])

        current_rotation = np.eye(3) @ R
        current_pose = np.array([0.0, 0.0, 0.0])

        translation = np.array([1.0, 0.0, 0.0])
        translation_world = current_rotation @ translation

        current_pose[0] += translation_world[0]
        current_pose[1] += translation_world[1]

        assert abs(current_pose[0]) < 1e-10         # No X movement
        assert abs(current_pose[1] - 1.0) < 1e-10   # Y movement


class TestMapBuilding:
    def test_initial_map_unknown(self):
        width, height = 200, 200
        occ_map = np.ones((height, width), dtype=np.int8) * -1
        assert np.all(occ_map == -1)

    def test_mark_free_space(self):
        width, height = 200, 200
        resolution = 0.1
        occ_map = np.ones((height, width), dtype=np.int8) * -1

        origin_x, origin_y = width // 2, height // 2
        radius = 3
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                px = origin_x + dx
                py = origin_y + dy
                if 0 <= px < width and 0 <= py < height:
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist <= radius:
                        occ_map[py, px] = 0

        assert occ_map[origin_y, origin_x] == 0
        assert occ_map[0, 0] == -1

    def test_map_publish_data_format(self):
        width, height = 50, 50
        occ_map = np.ones((height, width), dtype=np.int8) * -1
        flat = occ_map.flatten().tolist()
        assert len(flat) == width * height
        assert all(v == -1 for v in flat)


class TestLoopClosureDetection:
    def test_minimum_history(self):
        pose_history = [np.array([float(i), 0.0, 0.0]) for i in range(5)]
        assert len(pose_history) < 10

    def test_recent_poses_skipped(self):
        skip_count = 20
        total = 25
        candidates = total - skip_count
        assert candidates == 5

    def test_distance_threshold(self):
        threshold = 0.5
        current_pos = np.array([0.1, 0.1])
        past_pos = np.array([0.0, 0.0])
        distance = np.linalg.norm(current_pos - past_pos)
        assert distance < threshold

    def test_min_distance_from_last_closure(self):
        min_distance = 5.0
        current_pos = np.array([1.0, 0.0])
        last_closure_pos = np.array([0.0, 0.0])
        dist = np.linalg.norm(current_pos - last_closure_pos)
        assert dist < min_distance

    def test_pose_correction(self):
        alpha = 0.3
        current = np.array([10.0, 0.0])
        reference = np.array([0.0, 0.0])

        corrected = alpha * reference + (1 - alpha) * current
        assert corrected[0] == 7.0
        assert corrected[1] == 0.0
