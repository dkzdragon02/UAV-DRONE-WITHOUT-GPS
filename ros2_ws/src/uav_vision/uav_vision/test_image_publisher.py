#!/usr/bin/env python3
"""
Test Image Publisher - Tạo ảnh test để kiểm thử hệ thống
Không cần camera thật, tạo ảnh synthetic với pattern di chuyển
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time


class TestImagePublisher(Node):
    """Publish test images với pattern di chuyển"""
    
    def __init__(self):
        super().__init__('test_image_publisher')
        
        # Parameters
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('pattern', 'moving_circle')  # 'moving_circle', 'checkerboard', 'random'
        
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.fps = self.get_parameter('fps').value
        self.pattern = self.get_parameter('pattern').value
        
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        
        # Timer
        period = 1.0 / self.fps
        self.timer = self.create_timer(period, self.timer_callback)
        
        # Animation state
        self.frame_count = 0
        self.circle_x = self.width // 2
        self.circle_y = self.height // 2
        self.direction_x = 1
        self.direction_y = 1
        self.speed = 2
        
        self.get_logger().info(f'Test Image Publisher started: {self.width}x{self.height} @ {self.fps}fps, pattern: {self.pattern}')
    
    def timer_callback(self):
        """Tạo và publish test image"""
        # Tạo ảnh dựa trên pattern
        if self.pattern == 'moving_circle':
            image = self.create_moving_circle_image()
        elif self.pattern == 'checkerboard':
            image = self.create_checkerboard_image()
        elif self.pattern == 'random':
            image = self.create_random_image()
        else:
            image = self.create_moving_circle_image()
        
        # Convert to ROS Image message
        try:
            msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera'
            self.publisher.publish(msg)
            self.frame_count += 1
        except Exception as e:
            self.get_logger().error(f'Error publishing image: {e}')
    
    def create_moving_circle_image(self):
        """Tạo ảnh với vòng tròn di chuyển"""
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Background gradient
        for y in range(self.height):
            intensity = int(50 + (y / self.height) * 50)
            image[y, :] = [intensity, intensity, intensity]
        
        # Cập nhật vị trí vòng tròn
        self.circle_x += self.direction_x * self.speed
        self.circle_y += self.direction_y * self.speed
        
        # Đổi hướng khi chạm biên
        if self.circle_x < 50 or self.circle_x > self.width - 50:
            self.direction_x *= -1
        if self.circle_y < 50 or self.circle_y > self.height - 50:
            self.direction_y *= -1
        
        # Vẽ vòng tròn
        cv2.circle(image, (int(self.circle_x), int(self.circle_y)), 30, (0, 255, 0), -1)
        cv2.circle(image, (int(self.circle_x), int(self.circle_y)), 30, (255, 255, 255), 2)
        
        # Vẽ một số features để VO có thể track
        for i in range(10):
            x = int(self.circle_x + 100 * np.cos(i * np.pi / 5))
            y = int(self.circle_y + 100 * np.sin(i * np.pi / 5))
            if 0 < x < self.width and 0 < y < self.height:
                cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
        
        # Thêm text
        cv2.putText(image, f'Frame: {self.frame_count}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f'Pos: ({int(self.circle_x)}, {int(self.circle_y)})', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return image
    
    def create_checkerboard_image(self):
        """Tạo ảnh checkerboard với pattern di chuyển"""
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Checkerboard pattern
        square_size = 50
        offset_x = int(self.frame_count * 0.5) % square_size
        offset_y = int(self.frame_count * 0.3) % square_size
        
        for y in range(0, self.height, square_size):
            for x in range(0, self.width, square_size):
                if ((x + offset_x) // square_size + (y + offset_y) // square_size) % 2 == 0:
                    cv2.rectangle(image, (x, y), (x + square_size, y + square_size), (255, 255, 255), -1)
                else:
                    cv2.rectangle(image, (x, y), (x + square_size, y + square_size), (0, 0, 0), -1)
        
        return image
    
    def create_random_image(self):
        """Tạo ảnh với features ngẫu nhiên"""
        image = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        
        # Thêm một số features cố định để track
        for i in range(20):
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            cv2.circle(image, (x, y), 5, (255, 255, 255), -1)
        
        return image


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

