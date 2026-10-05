import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np


class TemplateMotionTest(Node):

    def __init__(self):
        super().__init__('template_motion_test')

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
            'Template motion test started.'
        )

    def image_callback(self, msg):

        image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='mono8'
        )

        self.frame += 1

        if self.previous is None:

            self.previous = image.copy()

            self.features = cv2.goodFeaturesToTrack(
                image,
                maxCorners=30,
                qualityLevel=0.01,
                minDistance=10,
                blockSize=7
            )

            if self.features is None:
                self.get_logger().error(
                    'No features detected.'
                )
                return

            self.features = self.features.reshape(-1, 2)

            self.get_logger().info(
                f'Frame 1 | Features: {len(self.features)}'
            )

            return

        motions = []

        h, w = self.previous.shape

        for x, y in self.features:

            x = int(round(x))
            y = int(round(y))

            template_radius = 7
            search_radius = 20

            x0 = x - template_radius
            x1 = x + template_radius + 1
            y0 = y - template_radius
            y1 = y + template_radius + 1

            sx0 = max(0, x - search_radius)
            sx1 = min(w, x + search_radius + 1)
            sy0 = max(0, y - search_radius)
            sy1 = min(h, y + search_radius + 1)

            if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                continue

            template = self.previous[y0:y1, x0:x1]
            search = image[sy0:sy1, sx0:sx1]

            result = cv2.matchTemplate(
                search,
                template,
                cv2.TM_CCOEFF_NORMED
            )

            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            new_x = sx0 + max_loc[0] + template_radius
            new_y = sy0 + max_loc[1] + template_radius

            dx = new_x - x
            dy = new_y - y

            motions.append(
                (
                    np.hypot(dx, dy),
                    dx,
                    dy,
                    max_val
                )
            )

        if motions:

            motions = np.array(motions)

            self.get_logger().info(
                f'Frame {self.frame} | '
                f'Templates: {len(motions)} | '
                f'MeanMotion: {np.mean(motions[:,0]):.3f} px | '
                f'MedianMotion: {np.median(motions[:,0]):.3f} px | '
                f'MeanDX: {np.mean(motions[:,1]):.3f} | '
                f'MeanDY: {np.mean(motions[:,2]):.3f} | '
                f'MeanScore: {np.mean(motions[:,3]):.3f}'
            )

        self.previous = image.copy()


def main(args=None):

    rclpy.init(args=args)

    node = TemplateMotionTest()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
