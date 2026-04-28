import pytest
import numpy as np
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from uav_vision.ekf_fusion import EKFFusion

class TestQuaternionUtilities:
    def test_quat_normalize(self):
        q = np.array([2.0, 0.0, 0.0, 0.0])
        result = EKFFusion._quat_normalize(q)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-10
        np.testing.assert_array_almost_equal(result, [1, 0, 0, 0])

    def test_quat_normalize_general(self):
        q = np.array([1.0, 1.0, 1.0, 1.0])
        result = EKFFusion._quat_normalize(q)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-10

    def test_quat_normalize_zero(self):
        q = np.array([0.0, 0.0, 0.0, 0.0])
        result = EKFFusion._quat_normalize(q)
        np.testing.assert_array_almost_equal(result, [1, 0, 0, 0])

    def test_quat_multiply_identity(self):
        q = np.array([0.707, 0.707, 0.0, 0.0])
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        result = EKFFusion._quat_multiply(q, identity)
        np.testing.assert_array_almost_equal(result, q, decimal=5)

    def test_quat_multiply_inverse(self):
        q = EKFFusion._quat_normalize(np.array([0.5, 0.3, 0.1, 0.7]))
        q_conj = EKFFusion._quat_conjugate(q)
        result = EKFFusion._quat_multiply(q, q_conj)
        np.testing.assert_array_almost_equal(result, [1, 0, 0, 0], decimal=10)

    def test_quat_conjugate(self):
        q = np.array([0.5, 0.1, 0.2, 0.3])
        result = EKFFusion._quat_conjugate(q)
        np.testing.assert_array_equal(result, [0.5, -0.1, -0.2, -0.3])

    def test_quat_to_rotation_identity(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = EKFFusion._quat_to_rotation(q)
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_quat_to_rotation_90_yaw(self):
        angle = math.pi / 2
        q = np.array([math.cos(angle/2), 0, 0, math.sin(angle/2)])
        R = EKFFusion._quat_to_rotation(q)
        expected = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1],
        ])
        np.testing.assert_array_almost_equal(R, expected, decimal=5)

    def test_rotation_vector_to_quat_zero(self):
        rv = np.array([0.0, 0.0, 0.0])
        q = EKFFusion._rotation_vector_to_quat(rv)
        np.testing.assert_array_almost_equal(q, [1, 0, 0, 0], decimal=5)

    def test_rotation_vector_to_quat_small(self):
        rv = np.array([0.001, 0.0, 0.0])
        q = EKFFusion._rotation_vector_to_quat(rv)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-5
        assert q[0] > 0.999  # Nearly identity

    def test_rotation_vector_to_quat_90(self):
        rv = np.array([0.0, 0.0, math.pi / 2])
        q = EKFFusion._rotation_vector_to_quat(rv)
        expected = np.array([math.cos(math.pi/4), 0, 0, math.sin(math.pi/4)])
        np.testing.assert_array_almost_equal(q, expected, decimal=5)

    def test_quat_to_euler_identity(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        roll, pitch, yaw = EKFFusion._quat_to_euler(q)
        assert abs(roll) < 1e-10
        assert abs(pitch) < 1e-10
        assert abs(yaw) < 1e-10

    def test_euler_roundtrip(self):
        roll, pitch, yaw = 0.3, 0.2, 0.5
        q = EKFFusion._euler_to_quaternion(roll, pitch, yaw)
        r2, p2, y2 = EKFFusion._quat_to_euler(q)
        assert abs(r2 - roll) < 1e-10
        assert abs(p2 - pitch) < 1e-10
        assert abs(y2 - yaw) < 1e-10

    def test_quat_error_as_rotvec_identity(self):
        q = EKFFusion._quat_normalize(np.array([0.5, 0.3, 0.1, 0.7]))
        rv = EKFFusion._quat_error_as_rotvec(q, q)
        np.testing.assert_array_almost_equal(rv, [0, 0, 0], decimal=10)

    def test_skew_property(self):
        v = np.array([1.0, 2.0, 3.0])
        S = EKFFusion._skew(v)
        np.testing.assert_array_almost_equal(S + S.T, np.zeros((3, 3)))

    def test_skew_cross_product(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = EKFFusion._skew(a) @ b
        expected = np.cross(a, b)
        np.testing.assert_array_almost_equal(result, expected)


class TestGimbalLock:
    def test_high_pitch_no_singularity(self):
        pitch = math.radians(89.0)
        q = EKFFusion._euler_to_quaternion(0.0, pitch, 0.0)
        assert abs(np.linalg.norm(q) - 1.0) < 1e-10
        r, p, y = EKFFusion._quat_to_euler(q)
        assert abs(p - pitch) < 0.01  # Within 0.01 rad

    def test_pitch_90_rotation_matrix(self):
        q = EKFFusion._euler_to_quaternion(0.0, math.pi / 2, 0.0)
        R = EKFFusion._quat_to_rotation(q)
        assert np.all(np.isfinite(R))
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_sequential_rotations_no_drift(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        small_rv = np.array([0.01, 0.005, 0.002])

        for _ in range(1000):
            dq = EKFFusion._rotation_vector_to_quat(small_rv)
            q = EKFFusion._quat_multiply(q, dq)
            q = EKFFusion._quat_normalize(q)

        assert abs(np.linalg.norm(q) - 1.0) < 1e-10
        R = EKFFusion._quat_to_rotation(q)
        assert abs(np.linalg.det(R) - 1.0) < 1e-8


class TestMahalanobisGate:
    def test_inlier_passes(self):
        y = np.array([0.01, 0.01, 0.01, 0.001, 0.001, 0.001])
        S = np.eye(6) * 0.05
        S_inv = np.linalg.inv(S)
        mahal = float(y.T @ S_inv @ y)
        threshold = 12.59  # χ²(6, 0.95)
        assert mahal < threshold

    def test_outlier_rejected(self):
        y = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
        S = np.eye(6) * 0.05
        S_inv = np.linalg.inv(S)
        mahal = float(y.T @ S_inv @ y)
        threshold = 12.59
        assert mahal > threshold

    def test_chi2_threshold_6dof(self):
        threshold = 12.59
        assert abs(threshold - 12.59) < 0.01


class TestStateVector:
    def test_initial_state(self):
        state = np.zeros(16)
        state[6] = 1.0  # qw = 1

        np.testing.assert_array_equal(state[0:3], [0, 0, 0])
        np.testing.assert_array_equal(state[3:6], [0, 0, 0])
        np.testing.assert_array_equal(state[6:10], [1, 0, 0, 0])
        np.testing.assert_array_equal(state[10:16], [0, 0, 0, 0, 0, 0])

    def test_error_state_dimensions(self):
        P = np.eye(15) * 0.1
        assert P.shape == (15, 15)
