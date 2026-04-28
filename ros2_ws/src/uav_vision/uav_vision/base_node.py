import rclpy
from rclpy.node import Node
from typing import Optional, Dict, Any
import time
import threading
import logging
from enum import Enum

class NodeState(Enum):
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"

class BaseNode(Node):
    def __init__(self, node_name: str, **kwargs):
        super().__init__(node_name, **kwargs)
        
        self._state = NodeState.UNINITIALIZED
        self._state_lock = threading.Lock()
        self._start_time = time.time()
        self._config: Optional[Dict[str, Any]] = None
        self._last_heartbeat = time.time()
        self._heartbeat_timeout = 5.0  
        self._error_count = 0
        self._max_errors = 10
        self._processing_times = []
        self._max_processing_history = 100
        self.get_logger().info(f"{node_name} base node initialized")
    
    def initialize(self) -> bool:
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
            import traceback
            self.get_logger().error(
                f"{self.get_name()} initialization error: {e}\n{traceback.format_exc()}"
            )
            return False
    
    def _initialize_impl(self) -> bool:
        return True
    
    def start(self) -> bool:
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
            import traceback
            self.get_logger().error(
                f"{self.get_name()} start error: {e}\n{traceback.format_exc()}"
            )
            return False
    
    def _start_impl(self) -> bool:
        return True
    
    def stop(self) -> bool:
        if self._state == NodeState.SHUTTING_DOWN:
            return True
        
        self._state = NodeState.SHUTTING_DOWN
        try:
            result = self._stop_impl()
            self.get_logger().info(f"{self.get_name()} stopped")
            return result
        except Exception as e:
            import traceback
            self.get_logger().error(
                f"{self.get_name()} stop error: {e}\n{traceback.format_exc()}"
            )
            return False
    
    def _stop_impl(self) -> bool:
        return True
    
    def cleanup(self):
        self._cleanup_impl()
    
    def _cleanup_impl(self):
        pass
    
    def is_healthy(self) -> bool:
        if self._state == NodeState.ERROR:
            return False
        
        if self._error_count >= self._max_errors:
            return False
        
        elapsed = time.time() - self._last_heartbeat
        if elapsed > self._heartbeat_timeout:
            return False
        
        return True
    
    def update_heartbeat(self):
        self._last_heartbeat = time.time()
    
    def record_error(self, error: Exception):
        self._error_count += 1
        self.get_logger().error(
            f"Error recorded ({self._error_count}/{self._max_errors}): {error}"
        )
        
        if self._error_count >= self._max_errors:
            self._state = NodeState.ERROR
            self.get_logger().error(
                f"{self.get_name()} reached max errors, entering ERROR state"
            )
    
    def reset_error_count(self):
        self._error_count = 0
    
    def track_processing_time(self, duration: float):
        self._processing_times.append(duration)
        if len(self._processing_times) > self._max_processing_history:
            self._processing_times.pop(0)
    
    def get_avg_processing_time(self) -> float:
        if not self._processing_times:
            return 0.0
        return sum(self._processing_times) / len(self._processing_times)
    
    def get_state(self) -> NodeState:
        return self._state
    
    def get_uptime(self) -> float:
        return time.time() - self._start_time
    
    def set_config(self, config: Dict[str, Any]):
        self._config = config
    
    def get_config(self, key: str, default: Any = None) -> Any:
        if self._config is None:
            return default
        return self._config.get(key, default)
    
    def destroy_node(self):
        self.cleanup()
        super().destroy_node()

