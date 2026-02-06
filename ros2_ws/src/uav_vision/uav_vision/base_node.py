#!/usr/bin/env python3
"""
Base Node Class for UAV Vision System
Provides common functionality for all ROS2 nodes
"""

import rclpy
from rclpy.node import Node
from typing import Optional, Dict, Any
import time
import logging
from enum import Enum


class NodeState(Enum):
    """Node state enumeration"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


class BaseNode(Node):
    """
    Base class for all UAV Vision ROS2 nodes.
    
    Provides common functionality:
    - Configuration management
    - Health monitoring
    - Error handling
    - State management
    - Logging utilities
    """
    
    def __init__(self, node_name: str, **kwargs):
        """
        Initialize base node.
        
        Args:
            node_name: Name of the ROS2 node
            **kwargs: Additional arguments passed to Node.__init__
        """
        super().__init__(node_name, **kwargs)
        
        # Node state
        self._state = NodeState.UNINITIALIZED
        self._state_lock = rclpy.executors.Executor()
        self._start_time = time.time()
        
        # Configuration
        self._config: Optional[Dict[str, Any]] = None
        
        # Health monitoring
        self._last_heartbeat = time.time()
        self._heartbeat_timeout = 5.0  # seconds
        self._error_count = 0
        self._max_errors = 10
        
        # Performance tracking
        self._processing_times = []
        self._max_processing_history = 100
        
        self.get_logger().info(f"{node_name} base node initialized")
    
    def initialize(self) -> bool:
        """
        Initialize the node. Override in subclasses.
        
        Returns:
            True if initialization successful, False otherwise
        """
        self._state = NodeState.INITIALIZING
        try:
            result = self._initialize_impl()
            if result:
                self._state = NodeState.READY
                self.get_logger().info(f"{self.get_name()} initialized successfully")
            else:
                self._state = NodeState.ERROR
                self.get_logger().error(f"{self.get_name()} initialization failed")
            return result
        except Exception as e:
            self._state = NodeState.ERROR
            self.get_logger().error(
                f"{self.get_name()} initialization error: {e}",
                exc_info=True
            )
            return False
    
    def _initialize_impl(self) -> bool:
        """
        Implementation of initialization. Override in subclasses.
        
        Returns:
            True if successful, False otherwise
        """
        return True
    
    def start(self) -> bool:
        """
        Start the node operation.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self._state != NodeState.READY:
            self.get_logger().warn(
                f"{self.get_name()} cannot start: state is {self._state.value}"
            )
            return False
        
        try:
            self._state = NodeState.RUNNING
            result = self._start_impl()
            if result:
                self.get_logger().info(f"{self.get_name()} started")
            else:
                self._state = NodeState.ERROR
            return result
        except Exception as e:
            self._state = NodeState.ERROR
            self.get_logger().error(
                f"{self.get_name()} start error: {e}",
                exc_info=True
            )
            return False
    
    def _start_impl(self) -> bool:
        """
        Implementation of start. Override in subclasses.
        
        Returns:
            True if successful, False otherwise
        """
        return True
    
    def stop(self) -> bool:
        """
        Stop the node operation.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if self._state == NodeState.SHUTTING_DOWN:
            return True
        
        self._state = NodeState.SHUTTING_DOWN
        try:
            result = self._stop_impl()
            self.get_logger().info(f"{self.get_name()} stopped")
            return result
        except Exception as e:
            self.get_logger().error(
                f"{self.get_name()} stop error: {e}",
                exc_info=True
            )
            return False
    
    def _stop_impl(self) -> bool:
        """
        Implementation of stop. Override in subclasses.
        
        Returns:
            True if successful, False otherwise
        """
        return True
    
    def cleanup(self):
        """Cleanup resources. Override in subclasses if needed."""
        self._cleanup_impl()
    
    def _cleanup_impl(self):
        """Implementation of cleanup. Override in subclasses."""
        pass
    
    def is_healthy(self) -> bool:
        """
        Check if node is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        if self._state == NodeState.ERROR:
            return False
        
        if self._error_count >= self._max_errors:
            return False
        
        elapsed = time.time() - self._last_heartbeat
        if elapsed > self._heartbeat_timeout:
            return False
        
        return True
    
    def update_heartbeat(self):
        """Update heartbeat timestamp."""
        self._last_heartbeat = time.time()
    
    def record_error(self, error: Exception):
        """
        Record an error.
        
        Args:
            error: The exception that occurred
        """
        self._error_count += 1
        self.get_logger().error(
            f"Error recorded ({self._error_count}/{self._max_errors}): {error}",
            exc_info=True
        )
        
        if self._error_count >= self._max_errors:
            self._state = NodeState.ERROR
            self.get_logger().error(
                f"{self.get_name()} reached max errors, entering ERROR state"
            )
    
    def reset_error_count(self):
        """Reset error count."""
        self._error_count = 0
    
    def track_processing_time(self, duration: float):
        """
        Track processing time for performance monitoring.
        
        Args:
            duration: Processing duration in seconds
        """
        self._processing_times.append(duration)
        if len(self._processing_times) > self._max_processing_history:
            self._processing_times.pop(0)
    
    def get_avg_processing_time(self) -> float:
        """
        Get average processing time.
        
        Returns:
            Average processing time in seconds
        """
        if not self._processing_times:
            return 0.0
        return sum(self._processing_times) / len(self._processing_times)
    
    def get_state(self) -> NodeState:
        """
        Get current node state.
        
        Returns:
            Current node state
        """
        return self._state
    
    def get_uptime(self) -> float:
        """
        Get node uptime.
        
        Returns:
            Uptime in seconds
        """
        return time.time() - self._start_time
    
    def set_config(self, config: Dict[str, Any]):
        """
        Set node configuration.
        
        Args:
            config: Configuration dictionary
        """
        self._config = config
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if self._config is None:
            return default
        return self._config.get(key, default)
    
    def destroy_node(self):
        """Override to ensure cleanup before destruction."""
        self.cleanup()
        super().destroy_node()

