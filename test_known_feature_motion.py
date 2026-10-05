import sys
import numpy as np
import cv2

sys.path.insert(
    0,
    "/home/dkzdragon02/UAV-WITHOUT-GPS/1.GPT/"
    "ros2_ws/build/uav_vision"
)

from uav_vision.camera_publisher import CameraPublisher


def main():

    print("=" * 60)
    print("KNOWN FEATURE IMAGE MOTION TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # Create publisher object without starting ROS node.
    # ---------------------------------------------------------

    obj = CameraPublisher.__new__(CameraPublisher)

    obj.width = 640
    obj.height = 480

    # ---------------------------------------------------------
    # Same camera pose functions used by publisher.
    # ---------------------------------------------------------

    # We need the methods and parameters used by CameraPublisher.
    # Initialize the required attributes manually.

    obj.fx = 500.0
    obj.fy = 500.0
    obj.cx = obj.width / 2.0
    obj.cy = obj.height / 2.0

    # Same deterministic world points.
    np.random.seed(42)

    obj.world_points = np.column_stack([
        np.random.uniform(-5.0, 5.0, 800),
        np.random.uniform(-4.0, 4.0, 800),
        np.random.uniform(2.5, 12.0, 800)
    ]).astype(np.float64)

    # ---------------------------------------------------------
    # Generate two camera poses.
    # ---------------------------------------------------------

    position0, R0, *_ = obj.camera_pose(0)
    position1, R1, *_ = obj.camera_pose(1)

    pixels0 = obj.project_points(
        position0,
        R0
    )

    pixels1 = obj.project_points(
        position1,
        R1
    )

    # ---------------------------------------------------------
    # Compare the SAME 3D point IDs.
    # ---------------------------------------------------------

    valid = (
        np.isfinite(pixels0[:, 0]) &
        np.isfinite(pixels0[:, 1]) &
        np.isfinite(pixels1[:, 0]) &
        np.isfinite(pixels1[:, 1])
    )

    old = pixels0[valid]
    new = pixels1[valid]

    displacement = new - old
    magnitude = np.linalg.norm(
        displacement,
        axis=1
    )

    print()
    print(f"Same visible points : {len(old)}")
    print(f"Mean motion         : {np.mean(magnitude):.3f} px")
    print(f"Median motion       : {np.median(magnitude):.3f} px")
    print(f"Min motion          : {np.min(magnitude):.3f} px")
    print(f"Max motion          : {np.max(magnitude):.3f} px")

    print()
    print("First 20 known feature motions:")
    print()

    for i in range(min(20, len(old))):

        x0, y0 = old[i]
        x1, y1 = new[i]

        dx, dy = displacement[i]

        motion = magnitude[i]

        print(
            f"{i:02d}: "
            f"({x0:8.3f}, {y0:8.3f}) -> "
            f"({x1:8.3f}, {y1:8.3f}) | "
            f"DX={dx:8.3f} "
            f"DY={dy:8.3f} | "
            f"M={motion:7.3f}"
        )

    # ---------------------------------------------------------
    # Create visual images containing the SAME feature IDs.
    # ---------------------------------------------------------

    image0 = np.zeros(
        (obj.height, obj.width),
        dtype=np.uint8
    )

    image1 = np.zeros(
        (obj.height, obj.width),
        dtype=np.uint8
    )

    # Draw only first 30 features so they are easy to inspect.
    for feature_id in range(
        min(30, len(pixels0))
    ):

        x0, y0 = pixels0[feature_id]
        x1, y1 = pixels1[feature_id]

        if not (
            np.isfinite(x0) and
            np.isfinite(y0) and
            np.isfinite(x1) and
            np.isfinite(y1)
        ):
            continue

        p0 = (
            int(round(x0)),
            int(round(y0))
        )

        p1 = (
            int(round(x1)),
            int(round(y1))
        )

        cv2.circle(
            image0,
            p0,
            4,
            255,
            -1
        )

        cv2.circle(
            image1,
            p1,
            4,
            255,
            -1
        )

    cv2.imwrite(
        "/tmp/known_features_frame0.png",
        image0
    )

    cv2.imwrite(
        "/tmp/known_features_frame1.png",
        image1
    )

    print()
    print("Saved:")
    print("  /tmp/known_features_frame0.png")
    print("  /tmp/known_features_frame1.png")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()