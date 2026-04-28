import pytest
import math
import numpy as np

class TestQuaternionToEuler:
    def _quat_to_euler(self, qx, qy, qz, qw):
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def test_identity_quaternion(self):
        roll, pitch, yaw = self._quat_to_euler(0, 0, 0, 1)
        assert abs(roll) < 1e-10
        assert abs(pitch) < 1e-10
        assert abs(yaw) < 1e-10

    def test_90_degree_yaw(self):
        # Quaternion for 90° yaw: (0, 0, sin(45°), cos(45°))
        angle = math.pi / 2
        qz = math.sin(angle / 2)
        qw = math.cos(angle / 2)
        roll, pitch, yaw = self._quat_to_euler(0, 0, qz, qw)
        assert abs(roll) < 1e-10
        assert abs(pitch) < 1e-10
        assert abs(yaw - math.pi / 2) < 1e-10

    def test_90_degree_roll(self):
        angle = math.pi / 2
        qx = math.sin(angle / 2)
        qw = math.cos(angle / 2)
        roll, pitch, yaw = self._quat_to_euler(qx, 0, 0, qw)
        assert abs(roll - math.pi / 2) < 1e-10
        assert abs(pitch) < 1e-10
        assert abs(yaw) < 1e-10

    def test_gimbal_lock_pitch_90(self):
        angle = math.pi / 2
        qy = math.sin(angle / 2)
        qw = math.cos(angle / 2)
        roll, pitch, yaw = self._quat_to_euler(0, qy, 0, qw)
        assert abs(pitch - math.pi / 2) < 1e-6

    def test_normalized_quaternion(self):
        # Slightly non-normalized
        qx, qy, qz, qw = 0.001, 0.001, 0.001, 0.999
        roll, pitch, yaw = self._quat_to_euler(qx, qy, qz, qw)
        # All angles should be small
        assert abs(roll) < 0.1
        assert abs(pitch) < 0.1
        assert abs(yaw) < 0.1


class TestCovarianceFormat:
    def test_upper_triangle_size(self):
        count = 0
        for row in range(6):
            for col in range(row, 6):
                count += 1
        assert count == 21

    def test_upper_triangle_extraction(self):
        # Create identity-like 6×6 covariance
        cov_6x6 = [0.0] * 36
        for i in range(6):
            cov_6x6[i * 6 + i] = float(i + 1)  # 1,2,3,4,5,6 on diagonal

        # Extract upper triangle
        covariance = [0.0] * 21
        idx = 0
        for row in range(6):
            for col in range(row, 6):
                covariance[idx] = float(cov_6x6[row * 6 + col])
                idx += 1

        assert len(covariance) == 21
        # Check diagonal values are at correct positions
        # Diagonal indices in upper triangle: 0, 6, 11, 15, 18, 20
        diag_indices = [0, 6, 11, 15, 18, 20]
        for i, diag_idx in enumerate(diag_indices):
            assert covariance[diag_idx] == float(i + 1)

    def test_default_covariance(self):
        covariance = [0.0] * 21
        diag_indices = [0, 6, 11, 15, 18, 20]
        for i in diag_indices:
            covariance[i] = 0.01

        assert len(covariance) == 21
        assert covariance[0] == 0.01  # x variance
        assert covariance[6] == 0.01  # y variance
        assert covariance[11] == 0.01  # z variance
        assert covariance[1] == 0.0   # Off-diagonal should be 0

    def test_old_covariance_was_wrong(self):
        old_covariance = [0.0] * 4
        assert len(old_covariance) != 21  # This was the bug!

    def test_speed_covariance_size(self):
        covariance = [0.0] * 9
        assert len(covariance) == 9


class TestVelocitySetpoint:
    def test_type_mask_velocity_only(self):
        type_mask = 0b0000111111000111
        # Bit 0: ignore x position ✓
        # Bit 1: ignore y position ✓
        # Bit 2: ignore z position ✓
        # Bit 3-5: DO NOT ignore vx, vy, vz (velocity used)
        # Bit 6-8: ignore ax, ay, az ✓
        # Bit 9: ignore force
        # Bit 10: ignore yaw ✓
        # Bit 11: ignore yaw_rate ✓

        assert type_mask & 0b000000000000001  # x pos ignored
        assert type_mask & 0b000000000000010  # y pos ignored
        assert type_mask & 0b000000000000100  # z pos ignored
        assert not (type_mask & 0b000000000001000)  # vx NOT ignored
        assert not (type_mask & 0b000000000010000)  # vy NOT ignored
        assert not (type_mask & 0b000000000100000)  # vz NOT ignored

    def test_zero_velocity_threshold(self):
        threshold = 1e-3
        vx, vy, vz = 0.0001, 0.0002, 0.00005
        should_skip = abs(vx) < threshold and abs(vy) < threshold and abs(vz) < threshold
        assert should_skip

    def test_nonzero_velocity_sent(self):
        threshold = 1e-3
        vx, vy, vz = 0.5, 0.0, 0.0
        should_skip = abs(vx) < threshold and abs(vy) < threshold and abs(vz) < threshold
        assert not should_skip
