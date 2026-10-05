import rclpy
import cv2
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class VisualPose1SecTest(Node):

    def __init__(self):
        super().__init__("visual_pose_1sec_test")

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            10
        )

        # =====================================================
        # Camera intrinsics
        # =====================================================

        self.K = np.array(
            [
                [500.0, 0.0, 320.0],
                [0.0, 500.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64
        )

        self.reference_image = None
        self.reference_points = None

        self.reference_frame = None
        self.frame_count = 0

        self.finished = False

    # =========================================================
    # Rotation matrix
    # =========================================================

    def rotation_matrix(self, roll, pitch, yaw):

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

    # =========================================================
    # Camera pose
    # =========================================================

    def camera_pose(self, t):

        camera_x = 0.12 * t
        camera_y = 0.025 * t
        camera_z = 0.02 * np.sin(0.05 * t)

        roll = (
            np.deg2rad(2.0) *
            np.sin(0.040 * t)
        )

        pitch = (
            np.deg2rad(3.0) *
            np.sin(0.050 * t)
        )

        yaw = (
            np.deg2rad(4.0) *
            np.sin(0.030 * t)
        )

        position = np.array([
            camera_x,
            camera_y,
            camera_z
        ])

        R_wc = self.rotation_matrix(
            roll,
            pitch,
            yaw
        )

        return position, R_wc

    # =========================================================
    # Ground truth relative pose
    #
    # x0 = R0.T @ (X - C0)
    # x1 = R1.T @ (X - C1)
    #
    # Therefore:
    #
    # x1 = R1.T @ R0 @ x0
    #      + R1.T @ (C0 - C1)
    #
    # This matches recoverPose:
    #
    # x1 = R @ x0 + t
    # =========================================================

    def ground_truth_relative_pose(self):

        position0, R0 = self.camera_pose(0.0)
        position1, R1 = self.camera_pose(1.0)

        R_gt = R1.T @ R0

        t_gt = R1.T @ (
            position0 - position1
        )

        return R_gt, t_gt

    # =========================================================
    # Image callback
    # =========================================================

    def image_callback(self, msg):

        if self.finished:
            return

        image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="mono8"
        )

        self.frame_count += 1

        # =====================================================
        # Find reference frame
        # =====================================================

        if self.reference_image is None:

            points = cv2.goodFeaturesToTrack(
                image,
                maxCorners=300,
                qualityLevel=0.001,
                minDistance=5,
                blockSize=7
            )

            if points is None:

                self.get_logger().warn(
                    f"Frame {self.frame_count}: "
                    "no features, trying next frame..."
                )

                return

            if len(points) < 8:

                self.get_logger().warn(
                    f"Frame {self.frame_count}: "
                    f"only {len(points)} features, "
                    "trying next frame..."
                )

                return

            self.reference_image = image.copy()

            self.reference_points = (
                points.astype(np.float32)
            )

            self.reference_frame = (
                self.frame_count
            )

            self.get_logger().info(
                f"Reference frame "
                f"{self.reference_frame}: "
                f"{len(points)} features"
            )

            return

        # =====================================================
        # Wait 20 frames ≈ 1 second
        # =====================================================

        frame_delta = (
            self.frame_count -
            self.reference_frame
        )

        if frame_delta < 20:
            return

        self.get_logger().info(
            f"Running LK from frame "
            f"{self.reference_frame} "
            f"to frame {self.frame_count}"
        )

        # =====================================================
        # LK optical flow
        # =====================================================

        points2, status, error = cv2.calcOpticalFlowPyrLK(
            self.reference_image,
            image,
            self.reference_points,
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

        if points2 is None:

            print("ERROR: LK returned None.")

            self.finished = True
            return

        status = status.reshape(-1)

        old_points = (
            self.reference_points.reshape(-1, 2)
        )

        new_points = (
            points2.reshape(-1, 2)
        )

        good_old = old_points[
            status == 1
        ]

        good_new = new_points[
            status == 1
        ]

        tracked = len(good_old)

        print()
        print("=" * 70)
        print("VISUAL POSE 1 SECOND TEST")
        print("=" * 70)

        print(
            f"Reference frame : "
            f"{self.reference_frame}"
        )

        print(
            f"Current frame   : "
            f"{self.frame_count}"
        )

        print(
            f"Initial features: "
            f"{len(old_points)}"
        )

        print(
            f"Tracked features: "
            f"{tracked}"
        )

        if tracked < 8:

            print(
                "ERROR: Not enough "
                "tracked features."
            )

            self.finished = True
            return

        # =====================================================
        # Image motion
        # =====================================================

        displacement = (
            good_new -
            good_old
        )

        motion = np.linalg.norm(
            displacement,
            axis=1
        )

        print(
            f"Mean image motion: "
            f"{np.mean(motion):.4f} px"
        )

        print(
            f"Median image motion: "
            f"{np.median(motion):.4f} px"
        )

        # =====================================================
        # Essential Matrix
        # =====================================================

        E, mask_E = cv2.findEssentialMat(
            good_old,
            good_new,
            self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None:

            print(
                "ERROR: Essential Matrix "
                "estimation failed."
            )

            self.finished = True
            return

        if mask_E is None:

            print(
                "ERROR: Essential Matrix "
                "mask is None."
            )

            self.finished = True
            return

        mask_E = mask_E.reshape(-1).astype(np.uint8)

        essential_inliers = int(
            np.count_nonzero(mask_E)
        )

        print(
            f"Essential inliers: "
            f"{essential_inliers}"
        )

        if essential_inliers < 8:

            print(
                "ERROR: Not enough "
                "Essential Matrix inliers."
            )

            self.finished = True
            return

        # =====================================================
        # Essential candidates
        # =====================================================

        if E.shape[0] % 3 != 0:

            print(
                "ERROR: Unexpected Essential "
                f"Matrix shape: {E.shape}"
            )

            self.finished = True
            return

        candidate_count = E.shape[0] // 3

        print(
            f"Essential candidates: "
            f"{candidate_count}"
        )

        # =====================================================
        # Recover pose
        #
        # IMPORTANT:
        # Pass the RANSAC mask directly to recoverPose().
        # =====================================================

        best_pose_inliers = -1
        best_R = None
        best_t = None
        best_candidate = None
        best_pose_mask = None

        for i in range(candidate_count):

            E_candidate = E[
                i * 3:
                (i + 1) * 3,
                :
            ]

            try:

                pose_inliers, R_est, t_est, pose_mask = (
                    cv2.recoverPose(
                        E_candidate,
                        good_old,
                        good_new,
                        self.K,
                        mask=mask_E.copy()
                    )
                )

            except TypeError:

                # Compatibility fallback for OpenCV builds
                # where the mask keyword behaves differently.

                essential_old = good_old[
                    mask_E.astype(bool)
                ]

                essential_new = good_new[
                    mask_E.astype(bool)
                ]

                pose_inliers, R_est, t_est, pose_mask = (
                    cv2.recoverPose(
                        E_candidate,
                        essential_old,
                        essential_new,
                        self.K
                    )
                )

            print(
                f"  Candidate {i + 1}: "
                f"{pose_inliers} pose inliers"
            )

            if pose_inliers > best_pose_inliers:

                best_pose_inliers = pose_inliers
                best_R = R_est
                best_t = t_est
                best_candidate = i + 1
                best_pose_mask = pose_mask

        pose_inliers = best_pose_inliers
        R_est = best_R
        t_est = best_t

        print()
        print(
            f"Best Essential candidate: "
            f"{best_candidate}"
        )

        print(
            f"Pose inliers     : "
            f"{pose_inliers}"
        )

        # =====================================================
        # Safety check
        # =====================================================

        if R_est is None or t_est is None:

            print(
                "ERROR: recoverPose() "
                "did not return a valid pose."
            )

            self.finished = True
            return

        # =====================================================
        # Estimated rotation
        # =====================================================

        print()
        print("Estimated R:")
        print(R_est)

        # =====================================================
        # Estimated translation direction
        # =====================================================

        t_norm = np.linalg.norm(
            t_est
        )

        if t_norm > 1e-12:

            t_direction = (
                t_est.reshape(3) /
                t_norm
            )

        else:

            t_direction = np.zeros(3)

        print()
        print(
            "Estimated translation direction:"
        )

        print(t_direction)

        # =====================================================
        # Ground truth
        # =====================================================

        R_gt, t_gt = (
            self.ground_truth_relative_pose()
        )

        t_gt_norm = np.linalg.norm(
            t_gt
        )

        if t_gt_norm > 1e-12:

            t_gt_direction = (
                t_gt /
                t_gt_norm
            )

        else:

            t_gt_direction = np.zeros(3)

        # =====================================================
        # Translation direction error
        # =====================================================

        dot_translation = np.dot(
            t_direction,
            t_gt_direction
        )

        dot_translation = np.clip(
            dot_translation,
            -1.0,
            1.0
        )

        translation_error = np.degrees(
            np.arccos(
                dot_translation
            )
        )

        # =====================================================
        # Rotation error
        # =====================================================

        R_error = (
            R_est @ R_gt.T
        )

        cos_angle = (
            np.trace(R_error) - 1.0
        ) / 2.0

        cos_angle = np.clip(
            cos_angle,
            -1.0,
            1.0
        )

        rotation_error = np.degrees(
            np.arccos(
                cos_angle
            )
        )

        # =====================================================
        # Print ground truth
        # =====================================================

        print()
        print("Ground truth relative R:")
        print(R_gt)

        print()
        print(
            "Ground truth translation direction:"
        )

        print(t_gt_direction)

        # =====================================================
        # Final result
        # =====================================================

        print()
        print("-" * 70)
        print("POSE RESULT")
        print("-" * 70)

        print(
            f"Rotation error           : "
            f"{rotation_error:.4f} deg"
        )

        print(
            f"Translation direction err: "
            f"{translation_error:.4f} deg"
        )

        print("-" * 70)

        if (
            pose_inliers >= 20 and
            rotation_error < 5.0 and
            translation_error < 20.0
        ):

            print(
                "POSE STATUS: PASS"
            )

        else:

            print(
                "POSE STATUS: FAIL"
            )

        print("-" * 70)

        self.finished = True


def main(args=None):

    rclpy.init(args=args)

    node = VisualPose1SecTest()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()