import sys
import numpy as np

sys.path.insert(
    0,
    "/home/dkzdragon02/UAV-WITHOUT-GPS/1.GPT/ros2_ws/build/uav_vision"
)

from uav_vision.camera_publisher import CameraPublisher


publisher = CameraPublisher.__new__(CameraPublisher)

publisher.width = 640
publisher.height = 480
publisher.fx = 500.0
publisher.fy = 500.0
publisher.cx = 320.0
publisher.cy = 240.0

np.random.seed(42)

publisher.world_points = np.column_stack([
    np.random.uniform(-5.0, 5.0, 800),
    np.random.uniform(-4.0, 4.0, 800),
    np.random.uniform(2.5, 12.0, 800)
]).astype(np.float64)


def get_pixels(t):

    position, R_wc, _, _, _ = publisher.camera_pose(t)

    pixels = publisher.project_points(
        position,
        R_wc
    )

    valid = (
        np.isfinite(pixels[:, 0]) &
        np.isfinite(pixels[:, 1]) &
        (pixels[:, 0] >= 0) &
        (pixels[:, 0] < publisher.width) &
        (pixels[:, 1] >= 0) &
        (pixels[:, 1] < publisher.height)
    )

    return pixels, valid


p0, valid0 = get_pixels(0.0)
p1, valid1 = get_pixels(1.0)

same = valid0 & valid1

p0 = p0[same]
p1 = p1[same]

motion = np.linalg.norm(
    p1 - p0,
    axis=1
)

print()
print("========================================")
print("RENDERER PIXEL MOTION TEST")
print("========================================")
print(f"Same visible points : {len(p0)}")
print(f"Mean motion         : {np.mean(motion):.3f} px")
print(f"Median motion       : {np.median(motion):.3f} px")
print(f"Min motion          : {np.min(motion):.3f} px")
print(f"Max motion          : {np.max(motion):.3f} px")
print()
print("First 10 points:")
print()

for i in range(min(10, len(p0))):

    x0, y0 = p0[i]
    x1, y1 = p1[i]

    print(
        f"{i:02d}: "
        f"({x0:8.3f}, {y0:8.3f}) -> "
        f"({x1:8.3f}, {y1:8.3f}) | "
        f"{motion[i]:7.3f} px"
    )