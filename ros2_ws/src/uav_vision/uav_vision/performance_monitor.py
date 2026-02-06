#!/usr/bin/env python3
"""
Performance Monitoring Node
Monitor performance metrics của hệ thống
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, String
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
import time
from collections import deque
import numpy as np
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import os

# Import new performance tools
from uav_vision.memory_profiler import MemoryMonitor, get_memory_usage
from uav_vision.profiler import FunctionProfiler


class PerformanceMonitor(Node):
    """Performance monitoring node"""
    
    def __init__(self):
        super().__init__('performance_monitor')
        
        # Parameters
        self.declare_parameter('update_rate', 1.0)  # Hz
        
        update_rate = self.get_parameter('update_rate').value
        
        # Metrics storage
        self.cpu_usage = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        self.node_stats = {}  # {node_name: {cpu, memory, messages}}
        
        # Enhanced monitoring
        self.memory_monitor = MemoryMonitor(check_interval=1.0 / update_rate)
        self.memory_monitor.start_monitoring()
        
        # Publishers
        self.cpu_pub = self.create_publisher(Float64, '/uav/performance/cpu', 10)
        self.memory_pub = self.create_publisher(Float64, '/uav/performance/memory', 10)
        self.diagnostics_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        
        # Timer
        self.monitor_timer = self.create_timer(1.0 / update_rate, self.monitor_performance)
        
        self.get_logger().info('Performance Monitor started')
    
    def monitor_performance(self):
        """Monitor system performance"""
        if not PSUTIL_AVAILABLE:
            # Fallback if psutil not available
            cpu_percent = 0.0
            memory_percent = 0.0
        else:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
        
        self.cpu_usage.append(cpu_percent)
        self.memory_usage.append(memory_percent)
        
        # Enhanced memory monitoring
        mem_info = self.memory_monitor.check_memory()
        memory_mb = mem_info.get('rss_mb', 0.0)
        
        # Publish metrics
        cpu_msg = Float64()
        cpu_msg.data = float(cpu_percent)
        self.cpu_pub.publish(cpu_msg)
        
        memory_msg = Float64()
        memory_msg.data = float(memory_percent)
        self.memory_pub.publish(memory_msg)
        
        # Check for memory anomalies
        if self.memory_monitor.detect_anomaly(threshold_percent=20.0):
            self.get_logger().warn(
                f"Memory anomaly detected: {memory_mb:.2f} MB"
            )
        
        # Diagnostics
        self.publish_diagnostics(cpu_percent, memory_percent, memory_mb)
    
    def publish_diagnostics(self, cpu: float, memory: float, memory_mb: float = 0.0):
        """Publish diagnostic messages"""
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()
        
        # CPU status
        cpu_status = DiagnosticStatus()
        cpu_status.name = "CPU Usage"
        cpu_status.level = DiagnosticStatus.OK if cpu < 80 else DiagnosticStatus.WARN
        cpu_status.message = f"CPU: {cpu:.1f}%"
        cpu_status.values.append(KeyValue(key="cpu_percent", value=f"{cpu:.2f}"))
        diag_array.status.append(cpu_status)
        
        # Memory status
        memory_status = DiagnosticStatus()
        memory_status.name = "Memory Usage"
        memory_status.level = DiagnosticStatus.OK if memory < 80 else DiagnosticStatus.WARN
        memory_status.message = f"Memory: {memory:.1f}% ({memory_mb:.1f} MB)"
        memory_status.values.append(KeyValue(key="memory_percent", value=f"{memory:.2f}"))
        memory_status.values.append(KeyValue(key="memory_mb", value=f"{memory_mb:.2f}"))
        
        # Memory trend
        trend = self.memory_monitor.get_memory_trend()
        if trend:
            memory_status.values.append(KeyValue(key="trend", value=trend.get('trend', 'unknown')))
            memory_status.values.append(KeyValue(key="avg_mb", value=f"{trend.get('avg_mb', 0):.2f}"))
        
        diag_array.status.append(memory_status)
        
        self.diagnostics_pub.publish(diag_array)
    
    def get_stats(self) -> dict:
        """Get performance statistics"""
        trend = self.memory_monitor.get_memory_trend()
        
        stats = {
            'cpu': {
                'current': self.cpu_usage[-1] if self.cpu_usage else 0.0,
                'avg': np.mean(self.cpu_usage) if self.cpu_usage else 0.0,
                'max': np.max(self.cpu_usage) if self.cpu_usage else 0.0,
            },
            'memory': {
                'current': self.memory_usage[-1] if self.memory_usage else 0.0,
                'avg': np.mean(self.memory_usage) if self.memory_usage else 0.0,
                'max': np.max(self.memory_usage) if self.memory_usage else 0.0,
            },
            'memory_detailed': trend if trend else {},
            'function_profiling': FunctionProfiler.get_summary()
        }
        return stats


def main(args=None):
    rclpy.init(args=args)
    node = PerformanceMonitor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

