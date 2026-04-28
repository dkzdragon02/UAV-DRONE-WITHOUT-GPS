import numpy as np
from typing import List, Tuple, Optional
from scipy.optimize import minimize
from scipy.interpolate import splrep, splev
import math

class TrajectoryOptimizer:
    def __init__(
        self,
        max_velocity: float = 2.0,
        max_acceleration: float = 1.0,
        max_jerk: float = 0.5
    ):

        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk

    def smooth_path(
        self,
        path: List[List[float]],
        smoothing_factor: float = 0.5
    ) -> List[List[float]]:

        if len(path) < 3:
            return path

        smoothed = [path[0]]        # Keep first point

        for i in range(1, len(path) - 1):
            prev = np.array(path[i - 1])
            curr = np.array(path[i])
            next_p = np.array(path[i + 1])

            smoothed_point = (
                (1 - smoothing_factor) * curr +
                (smoothing_factor / 2) * (prev + next_p)
            )
            smoothed.append(smoothed_point.tolist())
        smoothed.append(path[-1])   # Keep last point
        return smoothed

    def optimize_velocity_profile(
        self,
        path: List[List[float]],
        target_time: Optional[float] = None
    ) -> List[float]:

        if len(path) < 2:
            return [0.0]

        segment_lengths = []
        for i in range(len(path) - 1):
            p1 = np.array(path[i])
            p2 = np.array(path[i + 1])
            length = np.linalg.norm(p2 - p1)
            segment_lengths.append(length)

        min_times = []
        for length in segment_lengths:
            if length == 0:
                min_times.append(0.1)  # Small time for zero-length segments
            else:
                t_accel = self.max_velocity / self.max_acceleration
                dist_accel = 0.5 * self.max_acceleration * t_accel ** 2

                if length <= 2 * dist_accel:
                    t = 2 * math.sqrt(length / self.max_acceleration)
                else:
                    t_const = (length - 2 * dist_accel) / self.max_velocity
                    t = 2 * t_accel + t_const

                min_times.append(t)

        total_min_time = sum(min_times)

        if target_time is None or target_time < total_min_time:
            target_time = total_min_time
        scale_factor = target_time / total_min_time
        times = [t * scale_factor for t in min_times]
        velocities = []
        for length, time in zip(segment_lengths, times):
            if time > 0:
                velocities.append(length / time)
            else:
                velocities.append(0.0)

        return velocities

    def add_acceleration_constraints(
        self,
        path: List[List[float]],
        velocities: List[float]
    ) -> List[float]:

        if len(path) < 2 or len(velocities) < 1:
            return velocities

        adjusted = [velocities[0]]

        for i in range(1, len(velocities)):
            p1 = np.array(path[i])
            p2 = np.array(path[i + 1])
            distance = np.linalg.norm(p2 - p1)

            if distance == 0:
                adjusted.append(0.0)
                continue

            prev_vel = adjusted[-1]
            max_vel_change = self.max_acceleration * (distance / prev_vel) if prev_vel > 0 else self.max_velocity
            new_vel = min(
                velocities[i],
                prev_vel + max_vel_change,
                self.max_velocity
            )
            new_vel = max(new_vel, 0.0)

            adjusted.append(new_vel)

        return adjusted

    def optimize_trajectory(
        self,
        path: List[List[float]],
        target_time: Optional[float] = None,
        smooth: bool = True
    ) -> Tuple[List[List[float]], List[float]]:

        if smooth:
            optimized_path = self.smooth_path(path)
        else:
            optimized_path = path

        velocities = self.optimize_velocity_profile(optimized_path, target_time)    # Optimize velocity profile
        velocities = self.add_acceleration_constraints(optimized_path, velocities)  # Apply acceleration constraints

        return optimized_path, velocities

    def interpolate_path(
        self,
        path: List[List[float]],
        resolution: float = 0.1
    ) -> List[List[float]]:

        if len(path) < 2:
            return path

        path_array = np.array(path)
        distances = [0.0]
        for i in range(1, len(path)):
            dist = np.linalg.norm(path_array[i] - path_array[i - 1])
            distances.append(distances[-1] + dist)

        total_distance = distances[-1]
        interpolated = []
        current_dist = 0.0

        while current_dist <= total_distance:
            for i in range(len(distances) - 1):
                if distances[i] <= current_dist <= distances[i + 1]:
                    alpha = (current_dist - distances[i]) / (
                        distances[i + 1] - distances[i]
                    ) if distances[i + 1] > distances[i] else 0.0

                    point = path_array[i] + alpha * (path_array[i + 1] - path_array[i])
                    interpolated.append(point.tolist())
                    break

            current_dist += resolution
        if interpolated[-1] != path[-1]:
            interpolated.append(path[-1])

        return interpolated

    def _allocate_segment_times(
        self,
        waypoints: List[List[float]],
        total_time: Optional[float] = None,
    ) -> List[float]:
        n_segments = len(waypoints) - 1
        if n_segments <= 0:
            return []

        distances = []
        for i in range(n_segments):
            d = np.linalg.norm(
                np.array(waypoints[i + 1]) - np.array(waypoints[i])
            )
            distances.append(max(d, 0.01))  # Avoid zero-length segments

        min_times = []
        for d in distances:
            t_accel = self.max_velocity / self.max_acceleration
            d_accel = 0.5 * self.max_acceleration * t_accel ** 2
            if d <= 2 * d_accel:
                t = 2.0 * math.sqrt(d / self.max_acceleration)
            else:
                t = 2.0 * t_accel + (d - 2.0 * d_accel) / self.max_velocity
            min_times.append(max(t, 0.1))

        total_min = sum(min_times)

        if total_time is not None and total_time > total_min:
            scale = total_time / total_min
            return [t * scale for t in min_times]

        return min_times

    @staticmethod
    def _minimum_snap_1d(
        positions: List[float],
        times: List[float],
        v0: float = 0.0,
        vf: float = 0.0,
        a0: float = 0.0,
        af: float = 0.0,
    ) -> List[np.ndarray]:
        n = len(times)  # Number of segments
        order = 8       # 7th-order polynomial → 8 coefficients per segment
        n_coeffs = n * order

        if n == 0:
            return []

        constraints_A = []
        constraints_b = []

        def add_constraint(row, val):
            constraints_A.append(row)
            constraints_b.append(val)

        def poly_coeffs(T, deriv=0):
            c = np.zeros(order)
            if deriv == 0:
                for k in range(order):
                    c[k] = T ** k
            elif deriv == 1:
                for k in range(1, order):
                    c[k] = k * T ** (k - 1)
            elif deriv == 2:
                for k in range(2, order):
                    c[k] = k * (k - 1) * T ** (k - 2)
            elif deriv == 3:
                for k in range(3, order):
                    c[k] = k * (k - 1) * (k - 2) * T ** (k - 3)
            return c

        row = np.zeros(n_coeffs)
        row[0:order] = poly_coeffs(0.0, 0)
        add_constraint(row, positions[0])
        row = np.zeros(n_coeffs)
        row[0:order] = poly_coeffs(0.0, 1)
        add_constraint(row, v0)
        row = np.zeros(n_coeffs)
        row[0:order] = poly_coeffs(0.0, 2)
        add_constraint(row, a0)
        row = np.zeros(n_coeffs)
        row[0:order] = poly_coeffs(0.0, 3)
        add_constraint(row, 0.0)
        T_last = times[-1]
        offset_last = (n - 1) * order
        row = np.zeros(n_coeffs)
        row[offset_last:offset_last + order] = poly_coeffs(T_last, 0)
        add_constraint(row, positions[-1])
        row = np.zeros(n_coeffs)
        row[offset_last:offset_last + order] = poly_coeffs(T_last, 1)
        add_constraint(row, vf)
        row = np.zeros(n_coeffs)
        row[offset_last:offset_last + order] = poly_coeffs(T_last, 2)
        add_constraint(row, af)
        row = np.zeros(n_coeffs)
        row[offset_last:offset_last + order] = poly_coeffs(T_last, 3)
        add_constraint(row, 0.0)

        for i in range(n - 1):
            T_i = times[i]
            off_i = i * order
            off_j = (i + 1) * order

            row = np.zeros(n_coeffs)
            row[off_i:off_i + order] = poly_coeffs(T_i, 0)
            add_constraint(row, positions[i + 1])
            row = np.zeros(n_coeffs)
            row[off_j:off_j + order] = poly_coeffs(0.0, 0)
            add_constraint(row, positions[i + 1])
            row = np.zeros(n_coeffs)
            row[off_i:off_i + order] = poly_coeffs(T_i, 1)
            row[off_j:off_j + order] = -poly_coeffs(0.0, 1)
            add_constraint(row, 0.0)
            row = np.zeros(n_coeffs)
            row[off_i:off_i + order] = poly_coeffs(T_i, 2)
            row[off_j:off_j + order] = -poly_coeffs(0.0, 2)
            add_constraint(row, 0.0)
            row = np.zeros(n_coeffs)
            row[off_i:off_i + order] = poly_coeffs(T_i, 3)
            row[off_j:off_j + order] = -poly_coeffs(0.0, 3)
            add_constraint(row, 0.0)

        A_eq = np.array(constraints_A)
        b_eq = np.array(constraints_b)

        H = np.zeros((n_coeffs, n_coeffs))
        for seg in range(n):
            T = times[seg]
            off = seg * order
            for i_c in range(4, order):
                for j_c in range(4, order):
                    coeff = 1.0
                    for k in range(4):
                        coeff *= (i_c - k) * (j_c - k)
                    power = i_c + j_c - 8 + 1
                    H[off + i_c, off + j_c] += coeff * T ** power / power

        H += np.eye(n_coeffs) * 1e-8
        n_constraints = A_eq.shape[0]

        KKT = np.zeros((n_coeffs + n_constraints, n_coeffs + n_constraints))
        KKT[:n_coeffs, :n_coeffs] = H
        KKT[:n_coeffs, n_coeffs:] = A_eq.T
        KKT[n_coeffs:, :n_coeffs] = A_eq

        rhs = np.zeros(n_coeffs + n_constraints)
        rhs[n_coeffs:] = b_eq

        try:
            solution = np.linalg.solve(KKT, rhs)
        except np.linalg.LinAlgError:
            solution, _, _, _ = np.linalg.lstsq(KKT, rhs, rcond=None)

        coeffs = solution[:n_coeffs]

        segments = []
        for seg in range(n):
            off = seg * order
            segments.append(coeffs[off:off + order].copy())

        return segments

    def generate_minimum_snap_trajectory(
        self,
        waypoints: List[List[float]],
        total_time: Optional[float] = None,
        dt: float = 0.02,
        v0: Optional[List[float]] = None,
        vf: Optional[List[float]] = None,
    ) -> Tuple[List[List[float]], List[List[float]], List[List[float]], List[float]]:
        if len(waypoints) < 2:
            return waypoints, [[0, 0, 0]] * len(waypoints), [[0, 0, 0]] * len(waypoints), [0.0]

        if v0 is None:
            v0 = [0.0, 0.0, 0.0]
        if vf is None:
            vf = [0.0, 0.0, 0.0]

        times = self._allocate_segment_times(waypoints, total_time)
        n_segments = len(times)

        wp_array = np.array(waypoints)
        dim = wp_array.shape[1] if wp_array.ndim > 1 else 1

        all_coeffs = []
        for axis in range(dim):
            positions_1d = wp_array[:, axis].tolist()
            coeffs_1d = self._minimum_snap_1d(
                positions_1d, times,
                v0=v0[axis] if axis < len(v0) else 0.0,
                vf=vf[axis] if axis < len(vf) else 0.0,
            )
            all_coeffs.append(coeffs_1d)

        positions = []
        velocities = []
        accelerations = []
        timestamps = []

        total = sum(times)
        t = 0.0
        seg_start = 0.0
        seg_idx = 0

        while t <= total + dt * 0.5:
            while seg_idx < n_segments - 1 and t >= seg_start + times[seg_idx]:
                seg_start += times[seg_idx]
                seg_idx += 1

            tau = t - seg_start  # Local time within segment
            tau = max(0.0, min(tau, times[seg_idx]))

            pos = []
            vel = []
            acc = []
            for axis in range(dim):
                c = all_coeffs[axis][seg_idx]
                p = sum(c[k] * tau ** k for k in range(len(c)))
                v = sum(k * c[k] * tau ** (k - 1) for k in range(1, len(c)))
                a = sum(k * (k - 1) * c[k] * tau ** (k - 2) for k in range(2, len(c)))
                pos.append(float(p))
                vel.append(float(v))
                acc.append(float(a))

            positions.append(pos)
            velocities.append(vel)
            accelerations.append(acc)
            timestamps.append(t)

            t += dt

        return positions, velocities, accelerations, timestamps

    def generate_smooth_trajectory(
        self,
        waypoints: List[List[float]],
        total_time: Optional[float] = None,
        dt: float = 0.05,
    ) -> Tuple[List[List[float]], List[float]]:
        positions, velocities, _, timestamps = self.generate_minimum_snap_trajectory(
            waypoints, total_time=total_time, dt=dt
        )

        for i in range(len(velocities)):
            speed = np.linalg.norm(velocities[i])
            if speed > self.max_velocity:
                scale = self.max_velocity / speed
                velocities[i] = [v * scale for v in velocities[i]]

        return positions, timestamps
