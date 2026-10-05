import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class ImageMotionTest(Node):

    def __init__(self):
        super().__init__('image_motion_test')

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
            'Image motion test started.'
        )

    def image_callback(self, msg):

        image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='mono8'
        )

        self.frame += 1

        if self.previous is not None:

            diff = cv2.absdiff(
                self.previous,
                image
            )

            mean_diff = float(np.mean(diff))
            max_diff = int(np.max(diff))
            changed_pixels = int(
                np.count_nonzero(diff > 0)
            )

            total_pixels = image.shape[0] * image.shape[1]

            changed_percent = (
                100.0 *
                changed_pixels /
                total_pixels
            )

            self.get_logger().info(
                f'Frame: {self.frame} | '
                f'MeanDiff: {mean_diff:.6f} | '
                f'MaxDiff: {max_diff} | '
                f'ChangedPixels: {changed_percent:.3f}%'
            )

        self.previous = image.copy()


def main(args=None):

    rclpy.init(args=args)

    node = ImageMotionTest()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()