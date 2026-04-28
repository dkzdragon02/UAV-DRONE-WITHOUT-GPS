import pytest
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from uav_vision.px4_controller import PositionPID

class TestPositionPID:
    def test_initial_zero_error(self):
        pid = PositionPID(kp=1.0, ki=0.0, kd=0.0)
        error = np.array([0.0, 0.0, 0.0])
        output = pid.compute(error)
        np.testing.assert_array_almost_equal(output, [0, 0, 0])

    def test_proportional_response(self):
        pid = PositionPID(kp=2.0, ki=0.0, kd=0.0)
        error = np.array([1.0, 0.0, 0.0])
        output = pid.compute(error)
        assert output[0] > 0

    def test_max_output_clamping(self):
        pid = PositionPID(kp=10.0, ki=0.0, kd=0.0, max_output=0.5)
        error = np.array([5.0, 5.0, 5.0])
        output = pid.compute(error)
        speed = np.linalg.norm(output)
        assert speed <= 0.5 + 1e-6

    def test_integral_accumulation(self):
        pid = PositionPID(kp=0.0, ki=1.0, kd=0.0, max_output=10.0)
        error = np.array([1.0, 0.0, 0.0])
        pid.compute(error)
        time.sleep(0.05)
        pid.compute(error)
        time.sleep(0.05)
        output = pid.compute(error)

        assert output[0] > 0

    def test_antiwindup(self):
        pid = PositionPID(kp=0.0, ki=10.0, kd=0.0, max_integral=1.0, max_output=100.0)
        error = np.array([100.0, 0.0, 0.0])

        for _ in range(50):
            pid.compute(error)
            time.sleep(0.01)

        assert abs(pid.integral[0]) <= 1.0 + 1e-6

    def test_derivative_response(self):
        pid = PositionPID(kp=0.0, ki=0.0, kd=1.0, max_output=100.0)
        pid.compute(np.array([0.0, 0.0, 0.0]))
        time.sleep(0.05)
        output = pid.compute(np.array([1.0, 0.0, 0.0]))

        assert output[0] > 0

    def test_reset_clears_state(self):
        pid = PositionPID(kp=1.0, ki=1.0, kd=1.0)

        for _ in range(10):
            pid.compute(np.array([1.0, 1.0, 1.0]))
            time.sleep(0.01)

        pid.reset()

        np.testing.assert_array_equal(pid.integral, [0, 0, 0])
        np.testing.assert_array_equal(pid.prev_error, [0, 0, 0])
        assert pid.last_time is None

    def test_3d_response(self):
        pid = PositionPID(kp=1.0, ki=0.0, kd=0.0, max_output=10.0)
        error = np.array([1.0, -2.0, 0.5])
        output = pid.compute(error)

        assert output[0] > 0    # positive x error → positive x output
        assert output[1] < 0    # negative y error → negative y output
        assert output[2] > 0    # positive z error → positive z output

    def test_convergence(self):
        pid = PositionPID(kp=2.0, ki=0.1, kd=0.5, max_output=1.0)

        position = np.array([1.0, 0.5, -0.3])  # Start away from target
        target = np.array([0.0, 0.0, 0.0])

        for _ in range(200):
            error = target - position
            correction = pid.compute(error)
            position += correction * 0.01  # dt = 0.01s
            time.sleep(0.005)

        dist = np.linalg.norm(position - target)
        assert dist < 0.5, f"Position did not converge: distance={dist:.3f}"
