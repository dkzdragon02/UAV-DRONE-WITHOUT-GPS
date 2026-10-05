import rclpy
import cv2
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class OpticalFlow1SecTest(Node):

    def __init__(self):

        super().__init__("optical_flow_1sec_test")

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            10
        )

        self.frame_count = 0

        self.first_image = None
        self.first_points = None

        self.test_started = False
        self.finished = False

    def image_callback(self, msg):

        if self.finished:
            return

        image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="mono8"
        )

        self.frame_count += 1

        # -------------------------------------------------
        # Frame 1: detect initial features.
        # -------------------------------------------------

        if self.first_image is None:

            self.first_image = image.copy()

            points = cv2.goodFeaturesToTrack(
                image,
                maxCorners=300,
                qualityLevel=0.01,
                minDistance=10,
                blockSize=7
            )

            if points is None:

                self.get_logger().error(
                    "No features detected."
                )

                self.finished = True
                return

            self.first_points = points.astype(
                np.float32
            )

            self.get_logger().info(
                f"Initial frame: "
                f"{len(points)} features"
            )

            return

        # -------------------------------------------------
        # Wait 20 frames.
        #
        # 20 frames × 0.05 s = 1 second.
        # -------------------------------------------------

        if self.frame_count < 21:
            return

        self.get_logger().info(
            "Running LK from frame 1 "
            "to frame 21 (~1 second)"
        )

        points1, status, error = (
            cv2.calcOpticalFlowPyrLK(
                self.first_image,
                image,
                self.first_points,
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
        )

        if points1 is None:

            self.get_logger().error(
                "LK returned None."
            )

            self.finished = True
            return

        status = status.reshape(-1)

        old_points = self.first_points.reshape(
            -1, 2
        )

        new_points = points1.reshape(
            -1, 2
        )

        old_valid = old_points[status == 1]
        new_valid = new_points[status == 1]

        if len(old_valid) == 0:

            self.get_logger().error(
                "No tracked features."
            )

            self.finished = True
            return

        # -------------------------------------------------
        # Calculate motion.
        # -------------------------------------------------

        displacement = (
            new_valid -
            old_valid
        )

        motion = np.linalg.norm(
            displacement,
            axis=1
        )

        dx = displacement[:, 0]
        dy = displacement[:, 1]

        self.get_logger().info(
            f"Tracked: {len(old_valid)}"
        )

        self.get_logger().info(
            f"MeanDX: {np.mean(dx):.4f}"
        )

        self.get_logger().info(
            f"MeanDY: {np.mean(dy):.4f}"
        )

        self.get_logger().info(
            f"MeanMotion: {np.mean(motion):.4f} px"
        )

        self.get_logger().info(
            f"MedianMotion: "
            f"{np.median(motion):.4f} px"
        )

        self.get_logger().info(
            f"MaxMotion: {np.max(motion):.4f} px"
        )

        # -------------------------------------------------
        # Print first 20.
        # -------------------------------------------------

        print()
        print("=" * 70)
        print("1 SECOND OPTICAL FLOW TEST")
        print("=" * 70)

        for i in range(
            min(20, len(old_valid))
        ):

            dx_i = displacement[i, 0]
            dy_i = displacement[i, 1]
            motion_i = motion[i]

            print(
                f"{i:02d}: "
                f"DX={dx_i:8.3f} "
                f"DY={dy_i:8.3f} "
                f"M={motion_i:8.3f}"
            )

        print("=" * 70)

        self.finished = True


def main(args=None):

    rclpy.init(args=args)

    node = OpticalFlow1SecTest()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()