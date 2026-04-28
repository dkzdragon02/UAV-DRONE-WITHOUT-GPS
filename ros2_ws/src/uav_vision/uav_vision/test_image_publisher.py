import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class TestImagePublisher(Node):
    def __init__(self):
        super().__init__('test_image_publisher')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('pattern', 'terrain')
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.fps = self.get_parameter('fps').value
        self.pattern = self.get_parameter('pattern').value
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)

        period = 1.0 / self.fps
        self.timer = self.create_timer(period, self.timer_callback)
        self.frame_count = 0
        self._rng = np.random.RandomState(42)  # Deterministic for reproducibility
        self._terrain_bg = self._generate_terrain_background()
        self._indoor_bg = self._generate_indoor_background()
        self._checkerboard_bg = self._generate_checkerboard_background()
        self._landmarks = self._generate_landmarks(count=80)
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._pan_vx = 0.3   # pixels/frame horizontal drift
        self._pan_vy = 0.15  # pixels/frame vertical drift

        self.get_logger().info(
            f'Test Image Publisher started: {self.width}x{self.height} '
            f'@ {self.fps}fps, pattern: {self.pattern}'
        )

    def timer_callback(self):
        pattern_map = {
            'terrain': self.create_terrain_image,
            'moving_features': self.create_moving_features_image,
            'indoor': self.create_indoor_image,
            'checkerboard': self.create_checkerboard_image,
        }
        gen = pattern_map.get(self.pattern, self.create_terrain_image)
        image = gen()

        try:
            msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera'
            self.publisher.publish(msg)
            self.frame_count += 1
        except Exception as e:
            self.get_logger().error(f'Error publishing image: {e}')

    def _generate_terrain_background(self):
        bw, bh = self.width * 3, self.height * 3
        bg = np.zeros((bh, bw, 3), dtype=np.uint8)

        for _ in range(200):
            cx = self._rng.randint(0, bw)
            cy = self._rng.randint(0, bh)
            r = self._rng.randint(20, 120)
            color = tuple(int(c) for c in self._rng.randint(30, 200, size=3))
            cv2.circle(bg, (cx, cy), r, color, -1)

        bg = cv2.GaussianBlur(bg, (31, 31), 0)

        for _ in range(300):
            cx = self._rng.randint(10, bw - 10)
            cy = self._rng.randint(10, bh - 10)
            size = self._rng.randint(3, 12)
            bright = tuple(int(c) for c in self._rng.randint(150, 255, size=3))
            shape_type = self._rng.randint(0, 3)
            if shape_type == 0:
                cv2.circle(bg, (cx, cy), size, bright, -1)
            elif shape_type == 1:
                pts = np.array([
                    [cx, cy - size],
                    [cx - size, cy + size],
                    [cx + size, cy + size]
                ])
                cv2.fillPoly(bg, [pts], bright)
            else:
                cv2.rectangle(bg, (cx - size, cy - size),
                              (cx + size, cy + size), bright, -1)

        return bg

    def _generate_indoor_background(self):
        bw, bh = self.width * 3, self.height * 3
        bg = np.full((bh, bw, 3), 180, dtype=np.uint8)  # Light gray floor

        for _ in range(40):
            x = self._rng.randint(0, bw)
            y = self._rng.randint(0, bh)
            dx = self._rng.randint(-200, 200)
            dy = self._rng.randint(-200, 200)
            color = tuple(int(c) for c in self._rng.randint(60, 140, size=3))
            cv2.line(bg, (x, y), (x + dx, y + dy), color, self._rng.randint(2, 6))

        for _ in range(50):
            x = self._rng.randint(0, bw - 80)
            y = self._rng.randint(0, bh - 80)
            w = self._rng.randint(20, 80)
            h = self._rng.randint(20, 80)
            color = tuple(int(c) for c in self._rng.randint(40, 220, size=3))
            cv2.rectangle(bg, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(bg, (x, y), (x + w, y + h), (0, 0, 0), 1)

        for _ in range(100):
            cx = self._rng.randint(5, bw - 5)
            cy = self._rng.randint(5, bh - 5)
            cv2.drawMarker(bg, (cx, cy), (0, 0, 0), cv2.MARKER_CROSS, 8, 1)

        return bg

    def _generate_landmarks(self, count=80):
        landmarks = []
        for _ in range(count):
            x = self._rng.randint(20, self.width * 3 - 20)
            y = self._rng.randint(20, self.height * 3 - 20)
            size = self._rng.randint(4, 15)
            color = tuple(int(c) for c in self._rng.randint(100, 255, size=3))
            shape = self._rng.randint(0, 3)
            landmarks.append((x, y, size, color, shape))
        return landmarks

    def create_terrain_image(self):
        bh, bw = self._terrain_bg.shape[:2]
        max_x = bw - self.width
        max_y = bh - self.height

        t = self.frame_count
        self._pan_x = (max_x / 2) + (max_x / 3) * np.sin(t * 0.002)
        self._pan_y = (max_y / 2) + (max_y / 3) * np.cos(t * 0.0015)

        sx = int(np.clip(self._pan_x, 0, max_x))
        sy = int(np.clip(self._pan_y, 0, max_y))

        crop = self._terrain_bg[sy:sy + self.height, sx:sx + self.width].copy()
        cv2.putText(crop, f'F:{self.frame_count}', (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

        return crop

    def create_indoor_image(self):
        bh, bw = self._indoor_bg.shape[:2]
        max_x = bw - self.width
        max_y = bh - self.height

        t = self.frame_count
        px = (max_x / 2) + (max_x / 3) * np.sin(t * 0.003)
        py = (max_y / 2) + (max_y / 4) * np.cos(t * 0.002)

        sx = int(np.clip(px, 0, max_x))
        sy = int(np.clip(py, 0, max_y))

        return self._indoor_bg[sy:sy + self.height, sx:sx + self.width].copy()

    def create_moving_features_image(self):
        bw, bh = self.width * 3, self.height * 3
        bg = np.zeros((bh, bw, 3), dtype=np.uint8)

        for y in range(bh):
            v = int(40 + 60 * (y / bh))
            bg[y, :] = [v, v + 10, v + 20]

        for (lx, ly, sz, col, shp) in self._landmarks:
            if shp == 0:
                cv2.circle(bg, (lx, ly), sz, col, -1)
            elif shp == 1:
                pts = np.array([
                    [lx, ly - sz], [lx - sz, ly + sz], [lx + sz, ly + sz]
                ])
                cv2.fillPoly(bg, [pts], col)
            else:
                cv2.rectangle(bg, (lx - sz, ly - sz),
                              (lx + sz, ly + sz), col, -1)

        max_x = bw - self.width
        max_y = bh - self.height
        t = self.frame_count
        px = (max_x / 2) + (max_x / 3) * np.sin(t * 0.0025)
        py = (max_y / 2) + (max_y / 3) * np.cos(t * 0.002)

        sx = int(np.clip(px, 0, max_x))
        sy = int(np.clip(py, 0, max_y))

        return bg[sy:sy + self.height, sx:sx + self.width].copy()

    def _generate_checkerboard_background(self):
        bw, bh = self.width * 3, self.height * 3
        bg = np.zeros((bh, bw, 3), dtype=np.uint8)
        square_size = 50

        for row in range(0, bh, square_size):
            for col in range(0, bw, square_size):
                grid_r = row // square_size
                grid_c = col // square_size
                is_white = (grid_r + grid_c) % 2 == 0
                if is_white:
                    base = 200 + (grid_r * 7 + grid_c * 13) % 55
                    color = (base, base - (grid_c * 3) % 30, base - (grid_r * 5) % 30)
                else:
                    base = 20 + (grid_r * 11 + grid_c * 7) % 40
                    color = (base + (grid_r * 3) % 20, base, base + (grid_c * 5) % 20)
                cv2.rectangle(bg, (col, row),
                              (col + square_size, row + square_size), color, -1)

        for row in range(0, bh, square_size):
            for col in range(0, bw, square_size):
                grid_r = row // square_size
                grid_c = col // square_size
                # Unique ID per intersection
                uid = grid_r * 1000 + grid_c
                marker_color = (
                    80 + (uid * 37) % 175,
                    80 + (uid * 53) % 175,
                    80 + (uid * 71) % 175,
                )

                shape_type = uid % 4
                sz = 4 + uid % 5
                if shape_type == 0:
                    cv2.circle(bg, (col, row), sz, marker_color, -1)
                elif shape_type == 1:
                    pts = np.array([
                        [col, row - sz], [col - sz, row + sz], [col + sz, row + sz]
                    ])
                    cv2.fillPoly(bg, [pts], marker_color)
                elif shape_type == 2:
                    cv2.rectangle(bg, (col - sz, row - sz),
                                  (col + sz, row + sz), marker_color, -1)
                else:
                    cv2.drawMarker(bg, (col, row), marker_color,
                                   cv2.MARKER_DIAMOND, sz * 2, 2)

        noise = self._rng.randint(0, 15, (bh, bw, 3), dtype=np.uint8)
        bg = cv2.add(bg, noise)

        return bg

    def create_checkerboard_image(self):
        bh, bw = self._checkerboard_bg.shape[:2]
        max_x = bw - self.width
        max_y = bh - self.height

        t = self.frame_count
        px = (max_x / 2) + (max_x / 3) * np.sin(t * 0.002)
        py = (max_y / 2) + (max_y / 3) * np.cos(t * 0.0015)

        sx = int(np.clip(px, 0, max_x))
        sy = int(np.clip(py, 0, max_y))

        crop = self._checkerboard_bg[sy:sy + self.height, sx:sx + self.width].copy()
        cv2.putText(crop, f'F:{self.frame_count}', (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    cv2.LINE_AA)
        return crop

def main(args=None):
    rclpy.init(args=args)
    node = TestImagePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
