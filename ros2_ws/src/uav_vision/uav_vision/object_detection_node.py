import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import numpy as np

class ObjectDetectionNode(Node):
    
    def __init__(self):
        super().__init__('object_detection_node')
        self.declare_parameter('model_type', 'yolo')            # 'yolo', 'ssd', 'custom'
        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('nms_threshold', 0.4)
        self.model_type = self.get_parameter('model_type').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.bridge = CvBridge()
        self.model = None
        self.init_model()                                       # Initialize model
        
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/uav/object_detection/detections',
            10
        )
        
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        self.get_logger().info(f'Object Detection Node started (model: {self.model_type})')
    
    def init_model(self):
        if self.model_type == 'yolo':
            try:
                model_path = self.get_parameter('model_path').value
                if model_path:
                    self.model = cv2.dnn.readNetFromDarknet(model_path + '.cfg', model_path + '.weights')
                else:
                    self.get_logger().warn("YOLO model path not specified, using basic detection")
                    self.model = None
            except Exception as e:
                self.get_logger().error(f"Failed to load YOLO model: {e}")
                self.model = None
        else:
            self.get_logger().warn(f"Model type {self.model_type} not yet implemented")
            self.model = None
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            if self.model is not None:
                detections = self.detect_objects(cv_image)
            else:
                detections = self.simple_detection(cv_image)
            
            detection_array = Detection2DArray()
            detection_array.header.stamp = msg.header.stamp
            detection_array.header.frame_id = 'camera'
            
            for det in detections:
                detection_msg = Detection2D()
                detection_msg.bbox.center.x = float(det['center'][0])
                detection_msg.bbox.center.y = float(det['center'][1])
                detection_msg.bbox.size_x = float(det['size'][0])
                detection_msg.bbox.size_y = float(det['size'][1])
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.id = det.get('class_id', 0)
                hypothesis.score = float(det.get('confidence', 1.0))
                detection_msg.results.append(hypothesis)
                detection_array.detections.append(detection_msg)
            self.detection_pub.publish(detection_array)
            
        except Exception as e:
            self.get_logger().error(f'Error processing object detection: {e}')
    
    def detect_objects(self, image):
        if self.model_type == 'yolo' and self.model is not None:
            return self.detect_yolo(image)
        return []
    
    def detect_yolo(self, image):
        # TODO: Implement YOLO detection
        # Cần có YOLO weights và config
        return []
    
    def simple_detection(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:                  # Filter small blobs
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    'center': (x + w/2, y + h/2),
                    'size': (w, h),
                    'confidence': 0.5
                })
        
        return detections

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

