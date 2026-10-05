import cv2
import numpy as np


# ============================================================
# Camera intrinsics
# ============================================================

width = 640
height = 480

fx = 500.0
fy = 500.0
cx = 320.0
cy = 240.0

K = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0]
], dtype=np.float64)


# ============================================================
# Generate 3D world points
# ============================================================

np.random.seed(42)

world_points = np.column_stack([
    np.random.uniform(-5.0, 5.0, 1000),
    np.random.uniform(-4.0, 4.0, 1000),
    np.random.uniform(3.0, 12.0, 1000)
]).astype(np.float64)


# ============================================================
# Camera pose
#
# R_wc = camera -> world
# position = camera position in world coordinates
# ============================================================

def rotation_matrix(roll, pitch, yaw):

    cr = np.cos(roll)
    sr = np.sin(roll)

    cp = np.cos(pitch)
    sp = np.sin(pitch)

    cy = np.cos(yaw)
    sy = np.sin(yaw)

    Rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr]
    ])

    Ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp]
    ])

    Rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx


# ============================================================
# Two camera poses
# ============================================================

position_1 = np.array([
    0.0,
    0.0,
    0.0
])

position_2 = np.array([
    0.10,
    0.02,
    0.01
])

R_wc_1 = rotation_matrix(
    np.deg2rad(0.0),
    np.deg2rad(0.0),
    np.deg2rad(0.0)
)

R_wc_2 = rotation_matrix(
    np.deg2rad(1.0),
    np.deg2rad(1.5),
    np.deg2rad(2.0)
)


# ============================================================
# Project world points into camera
# ============================================================

def project_points(world_points, position, R_wc):

    R_cw = R_wc.T

    camera_points = (
        R_cw @ (
            world_points.T -
            position.reshape(3, 1)
        )
    ).T

    valid = camera_points[:, 2] > 0.1

    camera_points = camera_points[valid]

    pixels = np.zeros((len(camera_points), 2))

    pixels[:, 0] = (
        fx * camera_points[:, 0] /
        camera_points[:, 2]
    ) + cx

    pixels[:, 1] = (
        fy * camera_points[:, 1] /
        camera_points[:, 2]
    ) + cy

    return pixels, valid


points_1, valid_1 = project_points(
    world_points,
    position_1,
    R_wc_1
)

points_2, valid_2 = project_points(
    world_points,
    position_2,
    R_wc_2
)


# ============================================================
# Keep only points visible in BOTH frames
# ============================================================

valid_both = valid_1 & valid_2

points_1, _ = project_points(
    world_points[valid_both],
    position_1,
    R_wc_1
)

points_2, _ = project_points(
    world_points[valid_both],
    position_2,
    R_wc_2
)


# Keep points inside image
inside = (
    (points_1[:, 0] >= 0) &
    (points_1[:, 0] < width) &
    (points_1[:, 1] >= 0) &
    (points_1[:, 1] < height) &
    (points_2[:, 0] >= 0) &
    (points_2[:, 0] < width) &
    (points_2[:, 1] >= 0) &
    (points_2[:, 1] < height)
)

points_1 = points_1[inside]
points_2 = points_2[inside]


print()
print("============================================================")
print("DIRECT POSE GEOMETRY TEST")
print("============================================================")
print(f"Correspondences: {len(points_1)}")


# ============================================================
# Essential Matrix
# ============================================================

E, mask = cv2.findEssentialMat(
    points_1,
    points_2,
    K,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=1.0
)

if E is None:
    print("ERROR: Essential Matrix was not found.")
    raise SystemExit(1)


if E.shape[0] > 3:
    E = E[:3, :3]


essential_inliers = int(mask.sum())

print(f"Essential inliers: {essential_inliers}")


# ============================================================
# Recover pose
# ============================================================

inlier_mask = mask.ravel().astype(bool)

inlier_points_1 = points_1[inlier_mask]
inlier_points_2 = points_2[inlier_mask]

pose_inliers, R_est, t_est, pose_mask = cv2.recoverPose(
    E,
    inlier_points_1,
    inlier_points_2,
    K
)


print(f"Pose inliers: {pose_inliers}")


# ============================================================
# Ground truth relative pose
# ============================================================

R_gt = R_wc_2.T @ R_wc_1

t_gt = R_wc_2.T @ (
    position_1 - position_2
)

t_gt = t_gt / np.linalg.norm(t_gt)


# ============================================================
# Rotation error
# ============================================================

R_error = R_est @ R_gt.T

trace = np.trace(R_error)

cos_angle = np.clip(
    (trace - 1.0) / 2.0,
    -1.0,
    1.0
)

rotation_error = np.rad2deg(
    np.arccos(cos_angle)
)


# ============================================================
# Translation direction error
# ============================================================

t_est_norm = t_est.reshape(3)
t_est_norm /= np.linalg.norm(t_est_norm)

t_gt_norm = t_gt.reshape(3)

cos_t = np.clip(
    np.dot(t_est_norm, t_gt_norm),
    -1.0,
    1.0
)

translation_error = np.rad2deg(
    np.arccos(cos_t)
)


# ============================================================
# Print results
# ============================================================

print()
print("Ground Truth:")
print("R_gt:")
print(R_gt)

print()
print("t_gt:")
print(t_gt)

print()
print("Estimated:")
print("R_est:")
print(R_est)

print()
print("t_est:")
print(t_est_norm)

print()
print("============================================================")
print(f"Rotation error:    {rotation_error:.3f} deg")
print(f"Translation error: {translation_error:.3f} deg")
print("============================================================")
print()