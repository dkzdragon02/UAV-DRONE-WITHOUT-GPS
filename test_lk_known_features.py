import sys
import numpy as np
import cv2

sys.path.insert(
    0,
    "/home/dkzdragon02/UAV-WITHOUT-GPS/1.GPT/"
    "ros2_ws/build/uav_vision"
)

from uav_vision.camera_publisher import CameraPublisher


WIDTH = 640
HEIGHT = 480


def make_texture(seed):
    rng = np.random.default_rng(seed)

    texture = rng.integers(
        0,
        256,
        size=(15, 15),
        dtype=np.uint8
    )

    # Tạo pattern có cấu trúc rõ ràng.
    texture[7, :] = 255
    texture[:, 7] = 255

    return texture


def draw_features(image, pixels, feature_ids):
    radius = 7

    for feature_id in feature_ids:

        x, y = pixels[feature_id]

        if not np.isfinite(x):
            continue

        if not np.isfinite(y):
            continue

        cx = int(round(x))
        cy = int(round(y))

        if (
            cx - radius < 0 or
            cx + radius >= WIDTH or
            cy - radius < 0 or
            cy + radius >= HEIGHT
        ):
            continue

        texture = make_texture(feature_id)

        x0 = cx - radius
        x1 = cx + radius + 1

        y0 = cy - radius
        y1 = cy + radius + 1

        image[
            y0:y1,
            x0:x1
        ] = texture


def main():

    print("=" * 70)
    print("LK KNOWN FEATURE TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Create CameraPublisher object without ROS initialization.
    # ---------------------------------------------------------

    obj = CameraPublisher.__new__(CameraPublisher)

    obj.width = WIDTH
    obj.height = HEIGHT

    obj.fx = 500.0
    obj.fy = 500.0
    obj.cx = WIDTH / 2.0
    obj.cy = HEIGHT / 2.0

    np.random.seed(42)

    obj.world_points = np.column_stack([
        np.random.uniform(-5.0, 5.0, 800),
        np.random.uniform(-4.0, 4.0, 800),
        np.random.uniform(2.5, 12.0, 800)
    ]).astype(np.float64)

    # ---------------------------------------------------------
    # Ground-truth camera poses.
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
    # Select sparse, well-separated features.
    # ---------------------------------------------------------

    feature_ids = []

    for feature_id in range(
        len(pixels0)
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

        if (
            x0 < 15 or x0 >= WIDTH - 15 or
            y0 < 15 or y0 >= HEIGHT - 15
        ):
            continue

        if (
            x1 < 15 or x1 >= WIDTH - 15 or
            y1 < 15 or y1 >= HEIGHT - 15
        ):
            continue

        # Require enough distance from existing features.
        good = True

        for selected_id in feature_ids:

            sx, sy = pixels0[selected_id]

            distance = np.hypot(
                x0 - sx,
                y0 - sy
            )

            if distance < 30.0:
                good = False
                break

        if not good:
            continue

        feature_ids.append(feature_id)

        if len(feature_ids) >= 50:
            break

    print()
    print(
        f"Selected features : {len(feature_ids)}"
    )

    # ---------------------------------------------------------
    # Create clean synthetic images.
    # ---------------------------------------------------------

    image0 = np.zeros(
        (HEIGHT, WIDTH),
        dtype=np.uint8
    )

    image1 = np.zeros(
        (HEIGHT, WIDTH),
        dtype=np.uint8
    )

    draw_features(
        image0,
        pixels0,
        feature_ids
    )

    draw_features(
        image1,
        pixels1,
        feature_ids
    )

    # ---------------------------------------------------------
    # Known feature coordinates.
    # ---------------------------------------------------------

    points0 = np.array([
        pixels0[i]
        for i in feature_ids
    ], dtype=np.float32)

    points1_gt = np.array([
        pixels1[i]
        for i in feature_ids
    ], dtype=np.float32)

    # ---------------------------------------------------------
    # Run LK optical flow.
    # ---------------------------------------------------------

    points1_lk, status, error = cv2.calcOpticalFlowPyrLK(
        image0,
        image1,
        points0.reshape(-1, 1, 2),
        None,
        winSize=(31, 31),
        maxLevel=4,
        criteria=(
            cv2.TERM_CRITERIA_EPS |
            cv2.TERM_CRITERIA_COUNT,
            50,
            0.001
        )
    )

    print()
    print("LK RESULT")
    print("-" * 70)

    if points1_lk is None:
        print("LK returned None.")
        return

    status = status.reshape(-1)

    gt = points1_gt[status == 1]
    estimated = points1_lk.reshape(-1, 2)[status == 1]

    old = points0[status == 1]

    # ---------------------------------------------------------
    # Ground-truth displacement.
    # ---------------------------------------------------------

    gt_motion = np.linalg.norm(
        gt - old,
        axis=1
    )

    # ---------------------------------------------------------
    # LK displacement.
    # ---------------------------------------------------------

    lk_motion = np.linalg.norm(
        estimated - old,
        axis=1
    )

    # ---------------------------------------------------------
    # Position error.
    # ---------------------------------------------------------

    position_error = np.linalg.norm(
        estimated - gt,
        axis=1
    )

    print(
        f"Tracked features : {len(old)}"
    )

    print(
        f"GT mean motion   : "
        f"{np.mean(gt_motion):.3f} px"
    )

    print(
        f"LK mean motion   : "
        f"{np.mean(lk_motion):.3f} px"
    )

    print(
        f"GT median motion : "
        f"{np.median(gt_motion):.3f} px"
    )

    print(
        f"LK median motion : "
        f"{np.median(lk_motion):.3f} px"
    )

    print(
        f"Mean position error : "
        f"{np.mean(position_error):.3f} px"
    )

    print(
        f"Median position error : "
        f"{np.median(position_error):.3f} px"
    )

    # ---------------------------------------------------------
    # Print individual results.
    # ---------------------------------------------------------

    print()
    print("FIRST 20 FEATURES")
    print("-" * 70)

    for i in range(
        min(20, len(old))
    ):

        gt_dx, gt_dy = (
            gt[i] - old[i]
        )

        lk_dx, lk_dy = (
            estimated[i] - old[i]
        )

        err = position_error[i]

        print(
            f"{i:02d} | "
            f"GT "
            f"DX={gt_dx:7.2f} "
            f"DY={gt_dy:7.2f} | "
            f"LK "
            f"DX={lk_dx:7.2f} "
            f"DY={lk_dy:7.2f} | "
            f"ERR={err:6.2f}"
        )

    # ---------------------------------------------------------
    # Save images.
    # ---------------------------------------------------------

    cv2.imwrite(
        "/tmp/lk_known_frame0.png",
        image0
    )

    cv2.imwrite(
        "/tmp/lk_known_frame1.png",
        image1
    )

    print()
    print("Saved:")
    print("  /tmp/lk_known_frame0.png")
    print("  /tmp/lk_known_frame1.png")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()