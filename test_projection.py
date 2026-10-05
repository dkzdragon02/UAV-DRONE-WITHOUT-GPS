import sys
import numpy as np

sys.path.insert(
    0,
    "/home/dkzdragon02/UAV-WITHOUT-GPS/1.GPT/ros2_ws/build/uav_vision"
)

from uav_vision.camera_publisher import CameraPublisher


def project_points(publisher, t):
    position, R_wc, _, _, _ = publisher.camera_pose(t)

    R_cw = R_wc.T

    points_camera = (
        R_cw @ (
            publisher.world_points.T -
            position.reshape(3, 1)
        )
    ).T

    valid = points_camera[:, 2] > 0.1

    u = np.full(len(points_camera), np.nan)
    v = np.full(len(points_camera), np.nan)

    valid_points = points_camera[valid]

    u_valid = (
        publisher.fx *
        valid_points[:, 0] /
        valid_points[:, 2]
        + publisher.cx
    )

    v_valid = (
        publisher.fy *
        valid_points[:, 1] /
        valid_points[:, 2]
        + publisher.cy
    )

    valid_indices = np.where(valid)[0]

    u[valid_indices] = u_valid
    v[valid_indices] = v_valid

    inside = (
        np.isfinite(u) &
        np.isfinite(v) &
        (u >= 0) &
        (u < publisher.width) &
        (v >= 0) &
        (v < publisher.height)
    )

    return np.column_stack((u, v)), inside


publisher = CameraPublisher.__new__(CameraPublisher)

publisher.fx = 500.0
publisher.fy = 500.0
publisher.cx = 320.0
publisher.cy = 240.0
publisher.width = 640
publisher.height = 480

np.random.seed(42)

publisher.world_points = np.column_stack([
    np.random.uniform(-5.0, 5.0, 800),
    np.random.uniform(-4.0, 4.0, 800),
    np.random.uniform(2.5, 12.0, 800)
]).astype(np.float64)


p0, valid0 = project_points(publisher, 0.0)
p1, valid1 = project_points(publisher, 1.0)

same_points = valid0 & valid1

p0_same = p0[same_points]
p1_same = p1[same_points]

displacement = np.linalg.norm(
    p1_same - p0_same,
    axis=1
)

print("=" * 70)
print("DIRECT PROJECTION TEST - SAME 3D POINTS")
print("=" * 70)

print(f"World points       : {len(publisher.world_points)}")
print(f"Visible at t=0     : {np.sum(valid0)}")
print(f"Visible at t=1     : {np.sum(valid1)}")
print(f"Visible in BOTH    : {len(p0_same)}")

print()
print(f"Mean displacement  : {np.mean(displacement):.3f} px")
print(f"Median displacement: {np.median(displacement):.3f} px")
print(f"Min displacement   : {np.min(displacement):.3f} px")
print(f"Max displacement   : {np.max(displacement):.3f} px")

print()
print("First 20 SAME 3D points:")
print(
    "ID     "
    "t=0(x,y)              "
    "t=1(x,y)              "
    "motion"
)

same_indices = np.where(same_points)[0]

for i in range(min(20, len(same_indices))):

    idx = same_indices[i]

    print(
        f"{idx:3d}    "
        f"[{p0[idx,0]:8.3f}, {p0[idx,1]:8.3f}]   "
        f"[{p1[idx,0]:8.3f}, {p1[idx,1]:8.3f}]   "
        f"{displacement[i]:8.3f} px"
    )

print("=" * 70)