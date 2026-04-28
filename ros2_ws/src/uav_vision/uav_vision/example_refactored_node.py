import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from typing import Optional
import numpy as np
import time
from uav_vision.base_node import BaseNode, NodeState
from uav_vision.config_manager import ConfigManager, ConfigValidator
from uav_vision.error_handler import ErrorHandler, RetryStrategy, retry_on_error, safe_execute
from uav_vision.health_monitor import HealthMonitor, HealthStatus
from uav_vision.utils import Timer

class ExampleRefactoredNode(BaseNode): 
    def __init__(self):
        super().__init__('example_refactored_node')
        self.config = ConfigManager()
        self._setup_config()
        self.error_handler = ErrorHandler(
            max_retries=3,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )
        
        self.health_monitor = HealthMonitor(
            node_name=self.get_name(),
            heartbeat_timeout=5.0
        )
        self._setup_health_checks()
        self.processed_frames = 0
        self.processing_timer = Timer()
        self.get_logger().info("Example refactored node created")
    
    def _setup_config(self):
        self.config.register_entry(
            key='image_topic',
            default='/camera/image_raw',
            description='Input image topic',
            required=True
        )
        
        self.config.register_entry(
            key='output_topic',
            default='/example/odometry',
            description='Output odometry topic',
            required=True
        )
        
        self.config.register_entry(
            key='processing_rate',
            default=30.0,
            description='Processing rate in Hz',
            required=False,
            validator=lambda x: ConfigValidator.validate_positive(x) and x <= 60.0
        )
        
        self.config.load_from_ros2_params(self)
        if not self.config.validate():
            errors = self.config.get_validation_errors()
            self.get_logger().error(f"Configuration validation failed: {errors}")
            raise ValueError("Invalid configuration")
    
    def _setup_health_checks(self):
        self.health_monitor.register_health_check(
            name='heartbeat',
            check_func=self.is_heartbeat_healthy,
            critical=True
        )

        self.health_monitor.register_health_check(
            name='processing_rate',
            check_func=self._check_processing_rate,
            timeout=2.0
        )

        self.health_monitor.register_component('image_processor')
        self.health_monitor.start_monitoring()
    
    def _initialize_impl(self) -> bool:
        try:
            image_topic = self.config.get('image_topic')
            output_topic = self.config.get('output_topic')
            self.image_sub = self.create_subscription(
                Image,
                image_topic,
                self.image_callback,
                10
            )

            self.odom_pub = self.create_publisher(
                Odometry,
                output_topic,
                10
            )
            
            rate = self.config.get('processing_rate', 30.0)
            self.timer = self.create_timer(1.0 / rate, self.timer_callback)
            
            self.get_logger().info("Example node initialized successfully")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Initialization failed: {e}", exc_info=True)
            return False
    
    def _start_impl(self) -> bool:
        self.get_logger().info("Example node started")
        return True
    
    def _stop_impl(self) -> bool:
        self.get_logger().info("Example node stopped")
        return True
    
    def _cleanup_impl(self):
        self.health_monitor.stop_monitoring()
        self.get_logger().info("Example node cleaned up")
    
    def image_callback(self, msg: Image):
        self.update_heartbeat()
        self.health_monitor.update_heartbeat()

        result = safe_execute(
            func=lambda: self._process_image(msg),
            default_return=None,
            error_handler=self.error_handler,
            context="image_processing"
        )
        
        if result is not None:
            self.processed_frames += 1
            self.health_monitor.update_component_health(
                'image_processor',
                HealthStatus.HEALTHY,
                message=f"Processed {self.processed_frames} frames"
            )
    
    @retry_on_error(max_retries=2, base_delay=0.1)
    def _process_image(self, msg: Image) -> Optional[Odometry]:
        self.processing_timer.start()
        
        try:
            time.sleep(0.01)                    # Simulate processing time
            odom = Odometry()
            odom.header.stamp = self.get_clock().now().to_msg()
            odom.header.frame_id = 'odom'
            odom.child_frame_id = 'base_link'
            elapsed = self.processing_timer.elapsed()
            
            self.track_processing_time(elapsed)
            self.odom_pub.publish(odom)
            
            return odom
            
        except Exception as e:
            self.record_error(e)
            self.health_monitor.record_error(e)
            raise
    
    def timer_callback(self):
        self.update_heartbeat()                 # Update heartbeat
        health_status = self.health_monitor.get_overall_health()
        if health_status != HealthStatus.HEALTHY:
            self.get_logger().warn(
                f"Health status: {health_status.value}"
            )
    
    def _check_processing_rate(self) -> bool:
        if self.processed_frames == 0:
            return False
        avg_time = self.get_avg_processing_time()
        expected_rate = self.config.get('processing_rate', 30.0)
        max_time = 1.0 / expected_rate
        return avg_time < max_time
    
    def is_heartbeat_healthy(self) -> bool:
        return self.is_healthy()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = ExampleRefactoredNode()
        if not node.initialize():
            node.get_logger().error("Failed to initialize node")
            return
        
        if not node.start():
            node.get_logger().error("Failed to start node")
            return
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.stop()
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

