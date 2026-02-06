"""
Unit tests for BaseNode
Note: Some tests require ROS2 environment
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

# Try to import ROS2, skip tests if not available
try:
    import rclpy
    from rclpy.node import Node
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    Node = object  # Dummy class for type hints

from uav_vision.base_node import BaseNode, NodeState


@pytest.mark.skipif(not ROS2_AVAILABLE, reason="ROS2 not available")
class TestBaseNode:
    """Test suite for BaseNode (requires ROS2)"""
    
    def test_init(self):
        """Test initialization"""
        rclpy.init()
        try:
            node = BaseNode('test_node')
            assert node.get_name() == 'test_node'
            assert node.get_state() == NodeState.UNINITIALIZED
        finally:
            rclpy.shutdown()
    
    def test_initialize_success(self):
        """Test successful initialization"""
        rclpy.init()
        try:
            node = BaseNode('test_node')
            
            # Override _initialize_impl to return True
            node._initialize_impl = Mock(return_value=True)
            
            result = node.initialize()
            
            assert result is True
            assert node.get_state() == NodeState.READY
        finally:
            rclpy.shutdown()
    
    def test_initialize_failure(self):
        """Test failed initialization"""
        rclpy.init()
        try:
            node = BaseNode('test_node')
            
            # Override _initialize_impl to return False
            node._initialize_impl = Mock(return_value=False)
            
            result = node.initialize()
            
            assert result is False
            assert node.get_state() == NodeState.ERROR
        finally:
            rclpy.shutdown()
    
    def test_initialize_exception(self):
        """Test initialization with exception"""
        rclpy.init()
        try:
            node = BaseNode('test_node')
            
            # Override _initialize_impl to raise exception
            node._initialize_impl = Mock(side_effect=ValueError("Init error"))
            
            result = node.initialize()
            
            assert result is False
            assert node.get_state() == NodeState.ERROR
        finally:
            rclpy.shutdown()
    
    def test_start_success(self):
        """Test successful start"""
        rclpy.init()
        try:
            node = BaseNode('test_node')
            node._initialize_impl = Mock(return_value=True)
            node._start_impl = Mock(return_value=True)
            
            node.initialize()
            result = node.start()
            
            assert result is True
            assert node.get_state() == NodeState.RUNNING
        finally:
            rclpy.shutdown()
    
    def test_start_before_ready(self):
        """Test starting before ready state"""
        rclpy.init()
        try:
            node = BaseNode('test_node')
            
            result = node.start()
            
            assert result is False
            assert node.get_state() != NodeState.RUNNING
        finally:
            rclpy.shutdown()
    
    def test_stop(self):
        """Test stopping"""
        rclpy.init()
        try:
            node = BaseNode('test_node')
            node._stop_impl = Mock(return_value=True)
            
            result = node.stop()
            
            assert result is True
            assert node.get_state() == NodeState.SHUTTING_DOWN
        finally:
            rclpy.shutdown()
    
    def test_is_healthy(self):
        """Test health check"""
        rclpy.init()
        try:
            node = BaseNode('test_node')
            node.update_heartbeat()
            
            assert node.is_healthy() is True
        finally:
            rclpy.shutdown()
    
    def test_is_healthy_with_errors(self):
        """Test health check with too many errors"""
        rclpy.init()
        try:
            node = BaseNode('test_node')
            node._max_errors = 2
            
            node.record_error(ValueError("Error 1"))
            node.record_error(ValueError("Error 2"))
            
            assert node.is_healthy() is False
        finally:
            rclpy.shutdown()
    
    def test_update_heartbeat(self):
        """Test heartbeat update"""
        rclpy.init()
        try:
            node = BaseNode('test_node')
            initial_time = node._last_heartbeat
            
            time.sleep(0.1)
            node.update_heartbeat()
            
            assert node._last_heartbeat > initial_time
        finally:
            rclpy.shutdown()
    
    def test_record_error(self):
        """Test error recording"""
        rclpy.init()
        try:
            node = BaseNode('test_node')
            node._max_errors = 5
            
            error = ValueError("Test error")
            node.record_error(error)
            
            assert node._error_count == 1
        finally:
            rclpy.shutdown()
    
    def test_record_error_max_reached(self):
        """Test error recording when max reached"""
        rclpy.init()
        try:
            node = BaseNode('test_node')
            node._max_errors = 2
            
            node.record_error(ValueError("Error 1"))
            node.record_error(ValueError("Error 2"))
            
            assert node._error_count == 2
            assert node.get_state() == NodeState.ERROR
        finally:
            rclpy.shutdown()
    
    def test_track_processing_time(self):
        """Test processing time tracking"""
        rclpy.init()
        try:
            node = BaseNode('test_node')
            
            node.track_processing_time(0.1)
            node.track_processing_time(0.2)
            node.track_processing_time(0.3)
            
            avg = node.get_avg_processing_time()
            assert avg == pytest.approx(0.2, rel=0.1)
        finally:
            rclpy.shutdown()
    
    def test_get_uptime(self):
        """Test getting uptime"""
        rclpy.init()
        try:
            node = BaseNode('test_node')
            
            time.sleep(0.1)
            uptime = node.get_uptime()
            
            assert uptime >= 0.1
        finally:
            rclpy.shutdown()
    
    def test_config_management(self):
        """Test configuration management"""
        rclpy.init()
        try:
            node = BaseNode('test_node')
            
            config = {'key1': 'value1', 'key2': 42}
            node.set_config(config)
            
            assert node.get_config('key1') == 'value1'
            assert node.get_config('key2') == 42
            assert node.get_config('nonexistent', 'default') == 'default'
        finally:
            rclpy.shutdown()
    
    def test_cleanup(self):
        """Test cleanup"""
        rclpy.init()
        try:
            node = BaseNode('test_node')
            node._cleanup_impl = Mock()
            
            node.cleanup()
            
            node._cleanup_impl.assert_called_once()
        finally:
            rclpy.shutdown()


class TestBaseNodeWithoutROS2:
    """Test BaseNode functionality that doesn't require ROS2"""
    
    def test_node_state_enum(self):
        """Test NodeState enum"""
        assert NodeState.UNINITIALIZED.value == "uninitialized"
        assert NodeState.READY.value == "ready"
        assert NodeState.RUNNING.value == "running"
        assert NodeState.ERROR.value == "error"

