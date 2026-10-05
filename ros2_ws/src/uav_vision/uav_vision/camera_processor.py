import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

import cv2
import numpy as np


class CameraProcessor(Node):

    def __init__(self):
        super().__init__('camera_processor')

        # -------------------------------------------------
        # Camera image subscription
        # -------------------------------------------------

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # -------------------------------------------------
        # Ground Truth subscription
        # -------------------------------------------------

        self.gt_subscription = self.create_subscription(
            Float32MultiArray,
            '/camera/ground_truth_pose',
            self.ground_truth_callback,
            10
        )

        self.bridge = CvBridge()

        self.prev_gray = None
        self.prev_points = None
        self.frame_count = 0

        # -------------------------------------------------
        # Latest Ground Truth
        # -------------------------------------------------

        self.gt_relative_R = None
        self.gt_relative_t = None

        # -------------------------------------------------
        # Camera intrinsics
        # -------------------------------------------------

        self.fx = 500.0
        self.fy = 500.0
        self.cx = 320.0
        self.cy = 240.0

        self.K = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        self.get_logger().info(
            'Camera Processor started. '
            'Optical Flow + Essential Matrix + Pose + Ground Truth'
        )

    # -----------------------------------------------------
    # Ground Truth callback
    # -----------------------------------------------------

    def ground_truth_callback(self, msg):

        if len(msg.data) != 12:

            self.get_logger().warn(
                f'Invalid Ground Truth message size: '
                f'{len(msg.data)}'
            )

            return

        data = np.asarray(
            msg.data,
            dtype=np.float64
        )

        # -------------------------------------------------
        # First 9 values = rotation matrix
        # Last 3 values = translation direction
        # -------------------------------------------------

        self.gt_relative_R = (
            data[0:9]
            .reshape(3, 3)
        )

        self.gt_relative_t = (
            data[9:12]
        )

    # -----------------------------------------------------
    # Rotation matrix -> Euler angles
    # -----------------------------------------------------

    def rotation_to_euler(self, R):

        sy = np.sqrt(
            R[0, 0] * R[0, 0] +
            R[1, 0] * R[1, 0]
        )

        singular = sy < 1e-6

        if not singular:

            roll = np.arctan2(
                R[2, 1],
                R[2, 2]
            )

            pitch = np.arctan2(
                -R[2, 0],
                sy
            )

            yaw = np.arctan2(
                R[1, 0],
                R[0, 0]
            )

        else:

            roll = np.arctan2(
                -R[1, 2],
                R[1, 1]
            )

            pitch = np.arctan2(
                -R[2, 0],
                sy
            )

            yaw = 0.0

        return np.degrees(
            [roll, pitch, yaw]
        )

    # -----------------------------------------------------
    # Rotation error
    # -----------------------------------------------------

    def calculate_rotation_error(
        self,
        R_est,
        R_gt
    ):

        R_error = (
            R_est
            @
            R_gt.T
        )

        trace_value = np.trace(
            R_error
        )

        cos_angle = (
            trace_value - 1.0
        ) / 2.0

        cos_angle = np.clip(
            cos_angle,
            -1.0,
            1.0
        )

        angle = np.arccos(
            cos_angle
        )

        return np.degrees(
            angle
        )

    # -----------------------------------------------------
    # Translation direction error
    # -----------------------------------------------------

    def calculate_translation_error(
        self,
        t_est,
        t_gt
    ):

        t_est = np.asarray(
            t_est,
            dtype=np.float64
        ).flatten()

        t_gt = np.asarray(
            t_gt,
            dtype=np.float64
        ).flatten()

        norm_est = np.linalg.norm(
            t_est
        )

        norm_gt = np.linalg.norm(
            t_gt
        )

        if (
            norm_est < 1e-9
            or
            norm_gt < 1e-9
        ):

            return None

        t_est = (
            t_est
            /
            norm_est
        )

        t_gt = (
            t_gt
            /
            norm_gt
        )

        dot_product = np.dot(
            t_est,
            t_gt
        )

        dot_product = np.clip(
            dot_product,
            -1.0,
            1.0
        )

        angle = np.arccos(
            dot_product
        )

        return np.degrees(
            angle
        )

    # -----------------------------------------------------
    # Feature detection
    # -----------------------------------------------------

    def detect_features(self, gray):

        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )

        return points

    # -----------------------------------------------------
    # Image callback
    # -----------------------------------------------------

    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        self.frame_count += 1

        # -------------------------------------------------
        # First frame
        # -------------------------------------------------

        if self.prev_gray is None:

            self.prev_gray = gray

            self.prev_points = (
                self.detect_features(
                    gray
                )
            )

            if self.prev_points is not None:

                self.get_logger().info(
                    f'Frame: {self.frame_count} | '
                    f'Initial features: '
                    f'{len(self.prev_points)}'
                )

            return

        # -------------------------------------------------
        # Make sure features exist
        # -------------------------------------------------

        if (
            self.prev_points is None
            or
            len(self.prev_points) < 8
        ):

            self.prev_points = (
                self.detect_features(
                    self.prev_gray
                )
            )

            self.get_logger().warn(
                'Not enough features. '
                'Re-detecting features.'
            )

            return

        # -------------------------------------------------
        # Lucas-Kanade Optical Flow
        # -------------------------------------------------

        new_points, status, error = (
            cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                self.prev_points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(
                    cv2.TERM_CRITERIA_EPS |
                    cv2.TERM_CRITERIA_COUNT,
                    30,
                    0.01
                )
            )
        )

        if new_points is None:

            self.prev_gray = gray

            self.prev_points = (
                self.detect_features(
                    gray
                )
            )

            return

        # -------------------------------------------------
        # Select valid optical-flow points
        # -------------------------------------------------

        good_old = self.prev_points[
            status == 1
        ]

        good_new = new_points[
            status == 1
        ]

        if len(good_old) < 8:

            self.get_logger().warn(
                f'Frame: {self.frame_count} | '
                f'Not enough tracked points: '
                f'{len(good_old)}'
            )

            self.prev_gray = gray

            self.prev_points = (
                self.detect_features(
                    gray
                )
            )

            return

        old_pts = good_old.reshape(
            -1,
            2
        )

        new_pts = good_new.reshape(
            -1,
            2
        )

        # -------------------------------------------------
        # Optical Flow motion
        # -------------------------------------------------

        motion = (
            new_pts
            -
            old_pts
        )

        dx = motion[:, 0]
        dy = motion[:, 1]

        median_dx = np.median(
            dx
        )

        median_dy = np.median(
            dy
        )

        distance = np.sqrt(
            (dx - median_dx) ** 2 +
            (dy - median_dy) ** 2
        )

        inlier_mask = (
            distance < 8.0
        )

        flow_old = old_pts[
            inlier_mask
        ]

        flow_new = new_pts[
            inlier_mask
        ]

        flow_inliers = len(
            flow_old
        )

        # -------------------------------------------------
        # Essential Matrix
        # -------------------------------------------------

        pose_inliers = 0
        essential_inliers = 0
        recover_mask_inliers = 0

        R = None
        t = None
        euler = None

        rotation_error = None
        translation_error = None

        if flow_inliers >= 8:

            E, E_mask = cv2.findEssentialMat(
                flow_old,
                flow_new,
                self.K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0
            )

            if (
                E is not None
                and
                E_mask is not None
            ):

                # -------------------------------------------------
                # OpenCV can return multiple 3x3 solutions.
                # Use the first solution.
                # -------------------------------------------------

                if E.shape[0] > 3:

                    E = E[
                        0:3,
                        0:3
                    ]

                if E.shape == (3, 3):

                    E_mask = (
                        E_mask
                        .ravel()
                        .astype(bool)
                    )

                    essential_inliers = int(
                        np.count_nonzero(
                            E_mask
                        )
                    )

                    if essential_inliers >= 8:

                        essential_old = (
                            flow_old[E_mask]
                        )

                        essential_new = (
                            flow_new[E_mask]
                        )

                        # -------------------------------------------------
                        # Recover relative camera pose
                        # using Essential Matrix inliers
                        # -------------------------------------------------

                        pose_inliers, R, t, pose_mask = (
                            cv2.recoverPose(
                                E,
                                essential_old,
                                essential_new,
                                self.K
                            )
                        )

                        recover_mask_inliers = int(
                            np.count_nonzero(
                                pose_mask
                            )
                        )

                        # -------------------------------------------------
                        # Accept pose
                        # -------------------------------------------------

                        if pose_inliers >= 20:

                            euler = (
                                self.rotation_to_euler(
                                    R
                                )
                            )

                            t_norm = np.linalg.norm(
                                t
                            )

                            if t_norm > 1e-9:

                                t = (
                                    t.flatten()
                                    /
                                    t_norm
                                )

                                # -------------------------------------------------
                                # Compare Estimated pose with Ground Truth
                                # -------------------------------------------------

                                if (
                                    self.gt_relative_R
                                    is not None
                                    and
                                    self.gt_relative_t
                                    is not None
                                ):

                                    rotation_error = (
                                        self.calculate_rotation_error(
                                            R,
                                            self.gt_relative_R
                                        )
                                    )

                                    translation_error = (
                                        self.calculate_translation_error(
                                            t,
                                            self.gt_relative_t
                                        )
                                    )

                        else:

                            R = None
                            t = None
                            euler = None

        # -------------------------------------------------
        # Logging
        # -------------------------------------------------

        message = (
            f'Frame: {self.frame_count} | '
            f'Tracked: {len(old_pts)} | '
            f'FlowInliers: {flow_inliers} | '
            f'dx: {median_dx:.3f} px | '
            f'dy: {median_dy:.3f} px | '
            f'Motion: '
            f'{np.sqrt(median_dx**2 + median_dy**2):.3f} px | '
            f'EssentialInliers: {essential_inliers} | '
            f'PoseInliers: {pose_inliers} | '
            f'RecoverMaskInliers: {recover_mask_inliers}'
        )

        # -------------------------------------------------
        # Estimated pose
        # -------------------------------------------------

        if euler is not None:

            roll, pitch, yaw = euler

            message += (
                f' | RPY(deg): '
                f'[{roll:.3f}, '
                f'{pitch:.3f}, '
                f'{yaw:.3f}]'
            )

            message += (
                f' | t: '
                f'[{t[0]:.3f}, '
                f'{t[1]:.3f}, '
                f'{t[2]:.3f}]'
            )

            # -------------------------------------------------
            # Pose errors
            # -------------------------------------------------

            if rotation_error is not None:

                message += (
                    f' | RotationError: '
                    f'{rotation_error:.3f} deg'
                )

            if translation_error is not None:

                message += (
                    f' | TranslationError: '
                    f'{translation_error:.3f} deg'
                )

        else:

            message += (
                ' | Pose: REJECTED'
            )

            # -------------------------------------------------
            # Ground Truth availability
            # -------------------------------------------------

            if (
                self.gt_relative_R is not None
                and
                self.gt_relative_t is not None
            ):

                message += (
                    ' | GT: AVAILABLE'
                )

            else:

                message += (
                    ' | GT: WAITING'
                )

        self.get_logger().info(
            message
        )

        # -------------------------------------------------
        # Update tracking state
        # -------------------------------------------------

        self.prev_gray = gray

        if flow_inliers < 50:

            self.prev_points = (
                self.detect_features(
                    gray
                )
            )

        else:

            self.prev_points = (
                flow_new
                .reshape(-1, 1, 2)
                .astype(np.float32)
            )


def main(args=None):

    rclpy.init(
        args=args
    )

    node = CameraProcessor()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()