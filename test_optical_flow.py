import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np


class OpticalFlowTest(Node):

    def __init__(self):
        super().__init__('optical_flow_test')

        self.bridge = CvBridge()
        self.previous = None
        self.frame = 0

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.get_logger().info(
            'Raw Optical Flow test started.'
        )

    def image_callback(self, msg):

        image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='mono8'
        )

        self.frame += 1

        if self.previous is None:

            self.previous = image.copy()

            features = cv2.goodFeaturesToTrack(
                image,
                maxCorners=300,
                qualityLevel=0.01,
                minDistance=5,
                blockSize=7
            )

            if features is None:
                self.get_logger().error(
                    'No features detected.'
                )
                return

            self.points = features.astype(np.float32)

            self.get_logger().info(
                f'Frame 1 | Features: {len(self.points)}'
            )

            return

        old_points = self.points

        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.previous,
            image,
            old_points,
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

        if new_points is None:
            self.get_logger().error(
                f'Frame {self.frame} | LK returned None'
            )
            return

        status = status.reshape(-1)

        old_good = old_points[status == 1]
        new_good = new_points[status == 1]

        if len(old_good) == 0:
            self.get_logger().error(
                f'Frame {self.frame} | No valid tracks'
            )
            return

        displacement = (
            new_good.reshape(-1, 2) -
            old_good.reshape(-1, 2)
        )

        dx = displacement[:, 0]
        dy = displacement[:, 1]

        magnitude = np.linalg.norm(
            displacement,
            axis=1
        )

        self.get_logger().info(
            f'Frame {self.frame} | '
            f'Tracked: {len(old_good)} | '
            f'MeanDX: {np.mean(dx):.4f} | '
            f'MeanDY: {np.mean(dy):.4f} | '
            f'MeanMotion: {np.mean(magnitude):.4f} | '
            f'MaxMotion: {np.max(magnitude):.4f}'
        )

        self.previous = image.copy()

        self.points = new_good.reshape(-1, 1, 2)


def main(args=None):

    rclpy.init(args=args)

    node = OpticalFlowTest()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()