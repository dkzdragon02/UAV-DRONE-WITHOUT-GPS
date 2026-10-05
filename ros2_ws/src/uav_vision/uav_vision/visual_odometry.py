import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge


class CameraPublisher(Node):

    def __init__(self):
        super().__init__("camera_publisher")

        # =====================================================
        # Camera
        # =====================================================

        self.width = 640
        self.height = 480

        self.fx = 500.0
        self.fy = 500.0
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

        self.bridge = CvBridge()

        # =====================================================
        # Image publisher
        # =====================================================

        self.publisher = self.create_publisher(
            Image,
            "/camera/image_raw",
            10
        )

        # =====================================================
        # Ground Truth publisher
        #
        # DEBUG ONLY.
        #
        # visual_odometry MUST NOT use this data for:
        #
        # - Essential Matrix
        # - recoverPose
        # - translation
        # - rotation
        # - trajectory
        # =====================================================

        self.gt_feature_publisher = self.create_publisher(
            Float32MultiArray,
            "/camera/ground_truth_features",
            10
        )

        self.get_logger().info(
            "GT feature validator enabled (DEBUG ONLY)."
        )

        # =====================================================
        # Deterministic 3D world
        # =====================================================

        rng = np.random.default_rng(42)

        self.world_points = np.column_stack([
            rng.uniform(-20.0, 80.0, 12000),
            rng.uniform(-12.0, 12.0, 12000),
            rng.uniform(5.0, 25.0, 12000)
        ]).astype(np.float64)

        # =====================================================
        # Select sparse features over the full simulation horizon.
        #
        # IMPORTANT:
        # The camera moves continuously along +X. Selecting landmarks
        # only from t=0 and t=1 s causes the visible feature population
        # to collapse later in the run even though the detector is healthy.
        #
        # The selection below keeps the scene sparse while making sure
        # that enough landmarks are available at multiple points along
        # the trajectory. This affects only the synthetic test scene and
        # GT diagnostics; it does not alter VO estimation.
        # =====================================================

        self.feature_ids = self.select_sparse_features()

        # =====================================================
        # Pre-generate deterministic visual patterns
        #
        # Each feature_id gets its own pattern.
        #
        # The pattern is generated once and reused for every
        # frame, so the same physical landmark keeps the same
        # visual identity during the entire simulation.
        # =====================================================

        self.feature_patterns = {}

        for feature_id in self.feature_ids:
            self.feature_patterns[feature_id] = (
                self._get_feature_pattern(feature_id)
            )

        # =====================================================
        # Frame counter
        #
        # Every new camera_publisher process starts at frame 1.
        # =====================================================

        self.frame_index = 0

        # =====================================================
        # Ground-truth diagnostic flag
        #
        # Run the geometry test only once.
        # =====================================================

        self.gt_geometry_test_done = False

        # =====================================================
        # Timer
        #
        # 0.05 s = 20 Hz
        # =====================================================

        self.timer = self.create_timer(
            0.05,
            self.publish_image
        )

        self.get_logger().info(
            "Camera Publisher started. "
            f"Sparse synthetic 3D scene: "
            f"{len(self.feature_ids)} features"
        )

        self.get_logger().info(
            "Unique landmark visual patterns: ENABLED"
        )

    # =========================================================
    # Unique feature pattern
    # =========================================================

    def _get_feature_pattern(self, feature_id):
        """
        Generate a deterministic 7x7 binary visual pattern
        for each feature_id.

        The same feature_id always produces the same pattern.

        This function ONLY changes image appearance.
        It does NOT change:

            - 3D coordinates
            - projection coordinates
            - Ground Truth
            - camera motion
            - pose calculation
        """

        pattern = np.zeros(
            (7, 7),
            dtype=np.uint8
        )

        # -----------------------------------------------------
        # Deterministic pseudo-random seed
        # -----------------------------------------------------

        seed = (
            int(feature_id) * 1103515245
            + 12345
        ) & 0x7FFFFFFF

        # -----------------------------------------------------
        # Candidate locations around the center
        #
        # Coordinates are (x, y)
        # -----------------------------------------------------

        positions = [
            (1, 1),
            (1, 3),
            (1, 5),
            (2, 2),
            (2, 4),
            (3, 1),
            (3, 5),
            (4, 2),
            (4, 4),
            (5, 1),
            (5, 3),
            (5, 5)
        ]

        selected = []

        # -----------------------------------------------------
        # Select deterministic positions
        # -----------------------------------------------------

        attempts = 0

        while len(selected) < 4 and attempts < 30:

            seed = (
                seed * 1103515245
                + 12345
            ) & 0x7FFFFFFF

            index = (
                seed %
                len(positions)
            )

            if index not in selected:
                selected.append(index)

            attempts += 1

        # -----------------------------------------------------
        # Center point
        #
        # This is the actual landmark center.
        # -----------------------------------------------------

        pattern[3, 3] = 255

        # -----------------------------------------------------
        # Unique surrounding structure
        # -----------------------------------------------------

        for index in selected:

            x, y = positions[index]

            pattern[y, x] = 255

        # -----------------------------------------------------
        # Add deterministic asymmetric element
        # -----------------------------------------------------

        seed = (
            seed * 1103515245
            + 12345
        ) & 0x7FFFFFFF

        extra_index = (
            seed %
            len(positions)
        )

        x, y = positions[extra_index]

        pattern[y, x] = 255

        return pattern

    # =========================================================
    # Draw unique feature
    # =========================================================

    def _draw_unique_feature(
        self,
        image,
        x,
        y,
        feature_id
    ):
        """
        Draw one synthetic landmark using its unique
        feature-specific pattern.

        The floating-point projection coordinate (x, y)
        remains the Ground Truth coordinate.

        The visual pattern is centered around that location.
        """

        height, width = image.shape[:2]

        # -----------------------------------------------------
        # Ignore invalid coordinates
        # -----------------------------------------------------

        if not (
            np.isfinite(x)
            and np.isfinite(y)
        ):
            return

        # -----------------------------------------------------
        # Keep enough image margin for the 7x7 pattern
        # -----------------------------------------------------

        if (
            x < 4.0
            or x >= width - 4.0
            or y < 4.0
            or y >= height - 4.0
        ):
            return

        # -----------------------------------------------------
        # Integer base coordinate
        # -----------------------------------------------------

        x_base = int(
            np.floor(x)
        )

        y_base = int(
            np.floor(y)
        )

        # -----------------------------------------------------
        # Sub-pixel position
        # -----------------------------------------------------

        fx = x - x_base
        fy = y - y_base

        # -----------------------------------------------------
        # Unique pattern for this feature
        # -----------------------------------------------------

        pattern = self.feature_patterns.get(
            feature_id
        )

        if pattern is None:

            pattern = self._get_feature_pattern(
                feature_id
            )

            self.feature_patterns[feature_id] = (
                pattern
            )

        # -----------------------------------------------------
        # Render the 7x7 unique pattern
        # -----------------------------------------------------

        for py in range(7):

            for px in range(7):

                if pattern[py, px] == 0:
                    continue

                local_x = (
                    px - 3
                )

                local_y = (
                    py - 3
                )

                target_x = (
                    x_base
                    + local_x
                )

                target_y = (
                    y_base
                    + local_y
                )

                # Top-left
                if (
                    0 <= target_x < width
                    and 0 <= target_y < height
                ):

                    value = int(
                        255.0
                        * (1.0 - fx)
                        * (1.0 - fy)
                    )

                    if value > image[
                        target_y,
                        target_x
                    ]:

                        image[
                            target_y,
                            target_x
                        ] = value

                # Top-right
                if (
                    0 <= target_x + 1 < width
                    and 0 <= target_y < height
                ):

                    value = int(
                        255.0
                        * fx
                        * (1.0 - fy)
                    )

                    if value > image[
                        target_y,
                        target_x + 1
                    ]:

                        image[
                            target_y,
                            target_x + 1
                        ] = value

                # Bottom-left
                if (
                    0 <= target_x < width
                    and 0 <= target_y + 1 < height
                ):

                    value = int(
                        255.0
                        * (1.0 - fx)
                        * fy
                    )

                    if value > image[
                        target_y + 1,
                        target_x
                    ]:

                        image[
                            target_y + 1,
                            target_x
                        ] = value

                # Bottom-right
                if (
                    0 <= target_x + 1 < width
                    and 0 <= target_y + 1 < height
                ):

                    value = int(
                        255.0
                        * fx
                        * fy
                    )

                    if value > image[
                        target_y + 1,
                        target_x + 1
                    ]:

                        image[
                            target_y + 1,
                            target_x + 1
                        ] = value

        # -----------------------------------------------------
        # Strong center
        # -----------------------------------------------------

        center_x = int(
            round(x)
        )

        center_y = int(
            round(y)
        )

        if (
            0 <= center_x < width
            and 0 <= center_y < height
        ):

            image[
                center_y,
                center_x
            ] = 255

    # =========================================================
    # Publish Ground Truth feature map
    # =========================================================

    def publish_ground_truth_features(
        self,
        frame_id,
        pixels
    ):

        msg = Float32MultiArray()

        data = [
            float(frame_id)
        ]

        visible_count = 0

        for feature_id in self.feature_ids:

            x, y = pixels[feature_id]

            if not (
                np.isfinite(x)
                and np.isfinite(y)
            ):
                continue

            if (
                x < 0.0
                or x >= self.width
                or y < 0.0
                or y >= self.height
            ):
                continue

            data.extend([
                float(feature_id),
                float(x),
                float(y)
            ])

            visible_count += 1

        msg.data = data

        self.gt_feature_publisher.publish(
            msg
        )

        if frame_id % 20 == 0:

            self.get_logger().info(
                "GT FEATURE MAP | "
                f"frame={frame_id} | "
                f"visible={visible_count}"
            )

    # =========================================================
    # Camera rotation
    # =========================================================

    def rotation_matrix(
        self,
        roll,
        pitch,
        yaw
    ):

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
        ], dtype=np.float64)

        Ry = np.array([
            [cp, 0, sp],
            [0, 1, 0],
            [-sp, 0, cp]
        ], dtype=np.float64)

        Rz = np.array([
            [cy, -sy, 0],
            [sy, cy, 0],
            [0, 0, 1]
        ], dtype=np.float64)

        return Rz @ Ry @ Rx

    # =========================================================
    # Camera pose
    # =========================================================

    def camera_pose(self, t):

        camera_x = 0.50 * t

        camera_y = (
            0.20 *
            np.sin(0.30 * t)
        )

        camera_z = (
            0.50 +
            0.30 *
            np.sin(0.20 * t)
        )

        roll = (
            np.deg2rad(3.0) *
            np.sin(0.40 * t)
        )

        pitch = (
            np.deg2rad(5.0) *
            np.sin(0.30 * t)
        )

        yaw = (
            np.deg2rad(8.0) *
            np.sin(0.20 * t)
        )

        position = np.array([
            camera_x,
            camera_y,
            camera_z
        ], dtype=np.float64)

        R_wc = self.rotation_matrix(
            roll,
            pitch,
            yaw
        )

        return (
            position,
            R_wc,
            roll,
            pitch,
            yaw
        )

    # =========================================================
    # 3D -> 2D projection
    # =========================================================

    def project_points(
        self,
        camera_position,
        R_wc
    ):

        relative = (
            self.world_points -
            camera_position
        )

        camera_points = (
            R_wc.T @ relative.T
        ).T

        z = camera_points[:, 2]

        pixels = np.full(
            (
                len(camera_points),
                2
            ),
            np.nan,
            dtype=np.float64
        )

        valid = z > 0.1

        x = camera_points[
            valid,
            0
        ]

        y = camera_points[
            valid,
            1
        ]

        zz = z[valid]

        pixels[
            valid,
            0
        ] = (
            self.fx * x / zz
            + self.cx
        )

        pixels[
            valid,
            1
        ] = (
            self.fy * y / zz
            + self.cy
        )

        return pixels

    # =========================================================
    # Select sparse, stable features over the trajectory
    # =========================================================

    def select_sparse_features(self):

        # Camera x(t) = 0.5 * t.  Sample 0-80 s so the selected landmark
        # set remains useful well beyond the original 40 s test window.
        # A fixed small quota per sample distributes landmarks across the
        # whole trajectory instead of exhausting the 500-feature budget
        # near the beginning of the run.
        sample_times = np.linspace(0.0, 80.0, 41)

        selected = []
        selected_set = set()

        # Aim for a moderate number of visible landmarks at each sampled
        # time.  The scene remains sparse, but coverage is maintained.
        target_visible_per_sample = 12
        grid_cell_px = 35.0

        for sample_time in sample_times:

            position, R_wc, *_ = self.camera_pose(float(sample_time))

            pixels = self.project_points(
                position,
                R_wc
            )

            # Deterministic grid occupancy for this sample.
            occupied = set()
            added_this_sample = 0

            # Iterate in deterministic landmark order.
            for feature_id in range(len(self.world_points)):

                if feature_id in selected_set:
                    continue

                x, y = pixels[feature_id]

                if not (
                    np.isfinite(x)
                    and np.isfinite(y)
                ):
                    continue

                margin = 8

                if (
                    x < margin
                    or x >= self.width - margin
                    or y < margin
                    or y >= self.height - margin
                ):
                    continue

                cell = (
                    int(x / grid_cell_px),
                    int(y / grid_cell_px)
                )

                if cell in occupied:
                    continue

                occupied.add(cell)
                selected.append(feature_id)
                selected_set.add(feature_id)
                added_this_sample += 1

                if added_this_sample >= target_visible_per_sample:
                    break

        # Keep the publisher lightweight and deterministic.
        # The exact count can vary slightly with trajectory visibility.
        return selected[:500]


    def draw_features(
        self,
        image,
        pixels
    ):

        for feature_id in self.feature_ids:

            x, y = pixels[
                feature_id
            ]

            if not (
                np.isfinite(x)
                and np.isfinite(y)
            ):
                continue

            if (
                x < 5.0
                or x >= self.width - 5.0
                or y < 5.0
                or y >= self.height - 5.0
            ):
                continue

            self._draw_unique_feature(
                image,
                x,
                y,
                feature_id
            )

    # =========================================================
    # Ground Truth relative pose
    #
    # Convention:
    #
    #     X1 = R_gt @ X0 + t_gt
    #
    # =========================================================

    def calculate_ground_truth_relative_pose(
        self,
        position0,
        R0,
        position1,
        R1
    ):

        R_relative = (
            R1.T @ R0
        )

        translation_camera1 = (
            R1.T @ (
                position0 -
                position1
            )
        )

        return (
            R_relative,
            translation_camera1
        )

    # =========================================================
    # DEBUG:
    # Test Essential Matrix using EXACT GT correspondences
    # =========================================================

    def debug_ground_truth_geometry(self):

        t0 = 0.0
        t1 = 1.0

        position0, R0, *_ = (
            self.camera_pose(t0)
        )

        position1, R1, *_ = (
            self.camera_pose(t1)
        )

        pixels0 = self.project_points(
            position0,
            R0
        )

        pixels1 = self.project_points(
            position1,
            R1
        )

        points_old = []
        points_new = []

        for feature_id in self.feature_ids:

            x0, y0 = pixels0[
                feature_id
            ]

            x1, y1 = pixels1[
                feature_id
            ]

            if not (
                np.isfinite(x0)
                and np.isfinite(y0)
                and np.isfinite(x1)
                and np.isfinite(y1)
            ):
                continue

            points_old.append([
                x0,
                y0
            ])

            points_new.append([
                x1,
                y1
            ])

        if len(points_old) < 8:

            self.get_logger().warning(
                "GT geometry test: "
                "not enough points: "
                f"{len(points_old)}"
            )

            return

        points_old = np.asarray(
            points_old,
            dtype=np.float64
        )

        points_new = np.asarray(
            points_new,
            dtype=np.float64
        )

        K = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        R_gt, t_gt = (
            self.calculate_ground_truth_relative_pose(
                position0,
                R0,
                position1,
                R1
            )
        )

        t_gt = t_gt.reshape(3)

        t_gt_norm = np.linalg.norm(
            t_gt
        )

        if t_gt_norm > 1e-12:

            t_gt_direction = (
                t_gt /
                t_gt_norm
            )

        else:

            t_gt_direction = np.zeros(
                3,
                dtype=np.float64
            )

        self.get_logger().info(
            "=================================================="
        )

        self.get_logger().info(
            "GROUND TRUTH RELATIVE POSE TEST"
        )

        self.get_logger().info(
            "=================================================="
        )

        self.get_logger().info(
            f"GT time: "
            f"{t0:.3f}s -> "
            f"{t1:.3f}s"
        )

        self.get_logger().info(
            f"GT correspondences: "
            f"{len(points_old)}"
        )

        self.get_logger().info(
            "GT position0: "
            f"["
            f"{position0[0]: .6f}, "
            f"{position0[1]: .6f}, "
            f"{position0[2]: .6f}"
            f"]"
        )

        self.get_logger().info(
            "GT position1: "
            f"["
            f"{position1[0]: .6f}, "
            f"{position1[1]: .6f}, "
            f"{position1[2]: .6f}"
            f"]"
        )

        translation_world = (
            position1 -
            position0
        )

        baseline = np.linalg.norm(
            translation_world
        )

        self.get_logger().info(
            "GT camera motion in world: "
            f"["
            f"{translation_world[0]: .6f}, "
            f"{translation_world[1]: .6f}, "
            f"{translation_world[2]: .6f}"
            f"]"
        )

        self.get_logger().info(
            f"GT baseline: "
            f"{baseline:.6f} m"
        )

        self.get_logger().info(
            "GT t "
            "(recoverPose convention): "
            f"["
            f"{t_gt[0]: .6f}, "
            f"{t_gt[1]: .6f}, "
            f"{t_gt[2]: .6f}"
            f"]"
        )

        self.get_logger().info(
            "GT t direction: "
            f"["
            f"{t_gt_direction[0]: .6f}, "
            f"{t_gt_direction[1]: .6f}, "
            f"{t_gt_direction[2]: .6f}"
            f"]"
        )

        self.get_logger().info(
            "GT R:"
        )

        for row in R_gt:

            self.get_logger().info(
                "  ["
                + ", ".join(
                    f"{v:.8f}"
                    for v in row
                )
                + "]"
            )

        # =====================================================
        # Essential Matrix
        # =====================================================

        try:

            E, mask_E = cv2.findEssentialMat(
                points_old,
                points_new,
                K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0
            )

        except cv2.error as e:

            self.get_logger().error(
                "GT geometry test "
                "findEssentialMat error: "
                f"{e}"
            )

            return

        if E is None:

            self.get_logger().warning(
                "GT geometry test: "
                "findEssentialMat returned None"
            )

            return

        if mask_E is not None:

            essential_inliers = int(
                np.count_nonzero(mask_E)
            )

        else:

            essential_inliers = 0

        # =====================================================
        # Parse Essential Matrix candidates
        # =====================================================

        if E.shape == (3, 3):

            E_candidates = [
                E
            ]

        elif (
            E.shape[1] == 3
            and E.shape[0] % 3 == 0
        ):

            E_candidates = [
                E[i:i + 3, :]
                for i in range(
                    0,
                    E.shape[0],
                    3
                )
            ]

        else:

            self.get_logger().warning(
                "GT geometry test: "
                f"unexpected Essential Matrix "
                f"shape: {E.shape}"
            )

            return

        # =====================================================
        # Recover pose
        # =====================================================

        best_pose_inliers = -1
        best_R = None
        best_t = None

        for E_index, E_candidate in enumerate(
            E_candidates
        ):

            try:

                pose_inliers, R, t, mask_pose = (
                    cv2.recoverPose(
                        E_candidate,
                        points_old,
                        points_new,
                        K
                    )
                )

            except cv2.error as e:

                self.get_logger().warning(
                    "GT geometry test "
                    f"recoverPose error "
                    f"for E[{E_index}]: "
                    f"{e}"
                )

                continue

            if pose_inliers > best_pose_inliers:

                best_pose_inliers = int(
                    pose_inliers
                )

                best_R = R.copy()
                best_t = t.copy()

        # =====================================================
        # Basic result
        # =====================================================

        self.get_logger().info(
            "=================================================="
        )

        self.get_logger().info(
            "ESSENTIAL / RECOVERPOSE RESULT"
        )

        self.get_logger().info(
            "=================================================="
        )

        self.get_logger().info(
            "GT GEOMETRY TEST: "
            f"points={len(points_old)} | "
            f"Essential={essential_inliers} | "
            f"RecoverPose={best_pose_inliers} | "
            f"E candidates={len(E_candidates)}"
        )

        if (
            best_R is None
            or best_t is None
        ):

            self.get_logger().warning(
                "GT geometry test: "
                "recoverPose failed to return "
                "a valid pose"
            )

            return

        # =====================================================
        # Normalize recovered translation
        # =====================================================

        t_recovered = (
            best_t.reshape(3)
        )

        t_recovered_norm = np.linalg.norm(
            t_recovered
        )

        if t_recovered_norm > 1e-12:

            t_recovered_direction = (
                t_recovered /
                t_recovered_norm
            )

        else:

            t_recovered_direction = np.zeros(
                3,
                dtype=np.float64
            )

        # =====================================================
        # Print recovered pose
        # =====================================================

        self.get_logger().info(
            "recoverPose t:"
        )

        self.get_logger().info(
            "  ["
            + ", ".join(
                f"{v:.8f}"
                for v in t_recovered
            )
            + "]"
        )

        self.get_logger().info(
            "recoverPose t direction:"
        )

        self.get_logger().info(
            "  ["
            + ", ".join(
                f"{v:.8f}"
                for v in t_recovered_direction
            )
            + "]"
        )

        self.get_logger().info(
            "recoverPose R:"
        )

        for row in best_R:

            self.get_logger().info(
                "  ["
                + ", ".join(
                    f"{v:.8f}"
                    for v in row
                )
                + "]"
            )

        # =====================================================
        # Compare translation direction
        # =====================================================

        dot_same = float(
            np.dot(
                t_gt_direction,
                t_recovered_direction
            )
        )

        dot_opposite = float(
            np.dot(
                t_gt_direction,
                -t_recovered_direction
            )
        )

        dot_same = np.clip(
            dot_same,
            -1.0,
            1.0
        )

        dot_opposite = np.clip(
            dot_opposite,
            -1.0,
            1.0
        )

        angle_same_deg = np.rad2deg(
            np.arccos(dot_same)
        )

        angle_opposite_deg = np.rad2deg(
            np.arccos(dot_opposite)
        )

        # =====================================================
        # Compare rotation
        # =====================================================

        R_error = (
            best_R.T @ R_gt
        )

        trace_value = float(
            np.trace(R_error)
        )

        cos_angle = (
            trace_value - 1.0
        ) / 2.0

        cos_angle = np.clip(
            cos_angle,
            -1.0,
            1.0
        )

        rotation_error_deg = np.rad2deg(
            np.arccos(cos_angle)
        )

        # =====================================================
        # Print comparison
        # =====================================================

        self.get_logger().info(
            "=================================================="
        )

        self.get_logger().info(
            "DIRECT GT vs recoverPose COMPARISON"
        )

        self.get_logger().info(
            "=================================================="
        )

        self.get_logger().info(
            "GT t direction:"
        )

        self.get_logger().info(
            "  ["
            + ", ".join(
                f"{v:.8f}"
                for v in t_gt_direction
            )
            + "]"
        )

        self.get_logger().info(
            "recoverPose t direction:"
        )

        self.get_logger().info(
            "  ["
            + ", ".join(
                f"{v:.8f}"
                for v in t_recovered_direction
            )
            + "]"
        )

        self.get_logger().info(
            "Translation angle "
            "GT vs recoverPose: "
            f"{angle_same_deg:.6f} deg"
        )

        self.get_logger().info(
            "Translation angle "
            "GT vs -recoverPose: "
            f"{angle_opposite_deg:.6f} deg"
        )

        self.get_logger().info(
            "Rotation error: "
            f"{rotation_error_deg:.6f} deg"
        )

        # =====================================================
        # Interpret translation
        # =====================================================

        if angle_same_deg < 5.0:

            translation_result = (
                "MATCH"
            )

        elif angle_opposite_deg < 5.0:

            translation_result = (
                "MATCH WITH SIGN AMBIGUITY"
            )

        else:

            translation_result = (
                "MISMATCH"
            )

        self.get_logger().info(
            "Translation comparison: "
            f"{translation_result}"
        )

        # =====================================================
        # Interpret rotation
        # =====================================================

        if rotation_error_deg < 1.0:

            rotation_result = (
                "MATCH"
            )

        elif rotation_error_deg < 5.0:

            rotation_result = (
                "CLOSE"
            )

        else:

            rotation_result = (
                "MISMATCH"
            )

        self.get_logger().info(
            "Rotation comparison: "
            f"{rotation_result}"
        )

        # =====================================================
        # Final diagnostic
        # =====================================================

        if (
            best_pose_inliers >=
            0.8 * len(points_old)
            and rotation_error_deg < 1.0
            and (
                angle_same_deg < 5.0
                or
                angle_opposite_deg < 5.0
            )
        ):

            self.get_logger().info(
                "=================================================="
            )

            self.get_logger().info(
                "GT POSE CHECK: PASS"
            )

            self.get_logger().info(
                "recoverPose() agrees with Ground Truth."
            )

            self.get_logger().info(
                "=================================================="
            )

        else:

            self.get_logger().warning(
                "=================================================="
            )

            self.get_logger().warning(
                "GT POSE CHECK: FAIL"
            )

            self.get_logger().warning(
                "recoverPose() does NOT agree "
                "with Ground Truth."
            )

            self.get_logger().warning(
                "=================================================="
            )

    # =========================================================
    # Publish image
    # =========================================================

    def publish_image(self):

        # =====================================================
        # Current simulation time
        # =====================================================

        t = (
            self.frame_index *
            0.05
        )

        position, R_wc, roll, pitch, yaw = (
            self.camera_pose(t)
        )

        pixels = self.project_points(
            position,
            R_wc
        )

        # =====================================================
        # Ground-truth geometry diagnostic
        #
        # Run exactly ONCE.
        # =====================================================

        if not self.gt_geometry_test_done:

            self.debug_ground_truth_geometry()

            self.gt_geometry_test_done = True

        # =====================================================
        # Create black image
        # =====================================================

        image = np.zeros(
            (
                self.height,
                self.width
            ),
            dtype=np.uint8
        )

        # =====================================================
        # Draw synthetic landmarks
        # =====================================================

        self.draw_features(
            image,
            pixels
        )

        # =====================================================
        # Current ROS frame ID
        # =====================================================

        frame_id = (
            self.frame_index + 1
        )

        # =====================================================
        # Publish Ground Truth
        #
        # DEBUG ONLY.
        # =====================================================

        self.publish_ground_truth_features(
            frame_id,
            pixels
        )

        # =====================================================
        # Convert image to ROS Image
        # =====================================================

        msg = self.bridge.cv2_to_imgmsg(
            image,
            encoding="mono8"
        )

        msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        msg.header.frame_id = str(
            frame_id
        )

        self.publisher.publish(
            msg
        )

        # =====================================================
        # Next frame
        # =====================================================

        self.frame_index += 1


# =============================================================
# Main
# =============================================================

def main(args=None):

    rclpy.init(
        args=args
    )

    node = CameraPublisher()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()

