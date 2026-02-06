"""
Unit tests for HealthMonitor
"""

import pytest
import time
import threading

from uav_vision.health_monitor import (
    HealthMonitor,
    HealthStatus
)


class TestHealthMonitor:
    """Test suite for HealthMonitor"""
    
    def test_init(self):
        """Test initialization"""
        monitor = HealthMonitor('test_node', heartbeat_timeout=5.0)
        
        assert monitor.node_name == 'test_node'
        assert monitor.heartbeat_timeout == 5.0
        assert monitor._monitoring is False
    
    def test_register_health_check(self):
        """Test registering health check"""
        monitor = HealthMonitor('test_node')
        
        def check_func():
            return True
        
        monitor.register_health_check(
            name='test_check',
            check_func=check_func,
            timeout=5.0,
            critical=True
        )
        
        assert 'test_check' in monitor._health_checks
        check = monitor._health_checks['test_check']
        assert check.name == 'test_check'
        assert check.critical is True
        assert check.timeout == 5.0
    
    def test_register_component(self):
        """Test registering component"""
        monitor = HealthMonitor('test_node')
        monitor.register_component('test_component')
        
        assert 'test_component' in monitor._components
        component = monitor._components['test_component']
        assert component.name == 'test_component'
        assert component.status == HealthStatus.HEALTHY
    
    def test_update_heartbeat(self):
        """Test updating heartbeat"""
        monitor = HealthMonitor('test_node')
        initial_time = monitor._last_heartbeat
        
        time.sleep(0.1)
        monitor.update_heartbeat()
        
        assert monitor._last_heartbeat > initial_time
    
    def test_is_heartbeat_healthy(self):
        """Test heartbeat health check"""
        monitor = HealthMonitor('test_node', heartbeat_timeout=1.0)
        
        # Just updated, should be healthy
        monitor.update_heartbeat()
        assert monitor.is_heartbeat_healthy() is True
        
        # Wait longer than timeout
        time.sleep(1.1)
        assert monitor.is_heartbeat_healthy() is False
    
    def test_update_component_health(self):
        """Test updating component health"""
        monitor = HealthMonitor('test_node')
        monitor.register_component('test_component')
        
        monitor.update_component_health(
            'test_component',
            HealthStatus.UNHEALTHY,
            error_count=5,
            message='Test message'
        )
        
        component = monitor._components['test_component']
        assert component.status == HealthStatus.UNHEALTHY
        assert component.error_count == 5
        assert component.message == 'Test message'
    
    def test_get_overall_health_healthy(self):
        """Test getting overall health when healthy"""
        monitor = HealthMonitor('test_node')
        monitor.update_heartbeat()
        
        status = monitor.get_overall_health()
        assert status == HealthStatus.HEALTHY
    
    def test_get_overall_health_critical_heartbeat(self):
        """Test overall health with critical heartbeat failure"""
        monitor = HealthMonitor('test_node', heartbeat_timeout=0.1)
        
        # Wait for heartbeat timeout
        time.sleep(0.2)
        
        status = monitor.get_overall_health()
        assert status == HealthStatus.CRITICAL
    
    def test_get_overall_health_critical_component(self):
        """Test overall health with critical component"""
        monitor = HealthMonitor('test_node')
        monitor.update_heartbeat()
        
        monitor.register_component('critical_component')
        monitor.update_component_health(
            'critical_component',
            HealthStatus.CRITICAL
        )
        
        status = monitor.get_overall_health()
        assert status == HealthStatus.CRITICAL
    
    def test_get_overall_health_unhealthy_component(self):
        """Test overall health with unhealthy component"""
        monitor = HealthMonitor('test_node')
        monitor.update_heartbeat()
        
        monitor.register_component('unhealthy_component')
        monitor.update_component_health(
            'unhealthy_component',
            HealthStatus.UNHEALTHY
        )
        
        status = monitor.get_overall_health()
        assert status == HealthStatus.UNHEALTHY
    
    def test_get_overall_health_degraded_component(self):
        """Test overall health with degraded component"""
        monitor = HealthMonitor('test_node')
        monitor.update_heartbeat()
        
        monitor.register_component('degraded_component')
        monitor.update_component_health(
            'degraded_component',
            HealthStatus.DEGRADED
        )
        
        status = monitor.get_overall_health()
        assert status == HealthStatus.DEGRADED
    
    def test_run_health_checks_success(self):
        """Test running health checks with success"""
        monitor = HealthMonitor('test_node')
        
        check_called = [False]
        
        def check_func():
            check_called[0] = True
            return True
        
        monitor.register_health_check('test_check', check_func)
        monitor.run_health_checks()
        
        assert check_called[0] is True
        check = monitor._health_checks['test_check']
        assert check.last_result is True
        assert check.consecutive_failures == 0
    
    def test_run_health_checks_failure(self):
        """Test running health checks with failure"""
        monitor = HealthMonitor('test_node')
        
        def check_func():
            return False
        
        monitor.register_health_check('test_check', check_func, max_failures=2)
        monitor.run_health_checks()
        
        check = monitor._health_checks['test_check']
        assert check.last_result is False
        assert check.consecutive_failures == 1
    
    def test_run_health_checks_timeout(self):
        """Test health check timeout"""
        monitor = HealthMonitor('test_node')
        
        def slow_check():
            time.sleep(0.2)  # Longer than timeout
            return True
        
        monitor.register_health_check('slow_check', slow_check, timeout=0.1)
        monitor.run_health_checks()
        
        check = monitor._health_checks['slow_check']
        assert check.last_result is False
    
    def test_run_health_checks_exception(self):
        """Test health check with exception"""
        monitor = HealthMonitor('test_node')
        
        def failing_check():
            raise ValueError("Check failed")
        
        monitor.register_health_check('failing_check', failing_check)
        monitor.run_health_checks()
        
        check = monitor._health_checks['failing_check']
        assert check.last_result is False
        assert check.consecutive_failures == 1
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring"""
        monitor = HealthMonitor('test_node', check_interval=0.1)
        
        monitor.start_monitoring()
        assert monitor._monitoring is True
        assert monitor._monitor_thread is not None
        assert monitor._monitor_thread.is_alive()
        
        # Wait a bit
        time.sleep(0.2)
        
        monitor.stop_monitoring()
        assert monitor._monitoring is False
    
    def test_record_error(self):
        """Test recording errors"""
        monitor = HealthMonitor('test_node')
        
        error = ValueError("Test error")
        monitor.record_error(error)
        
        assert len(monitor._error_history) == 1
        error_record = monitor._error_history[0]
        assert error_record['type'] == 'ValueError'
        assert 'Test error' in error_record['error']
    
    def test_get_health_report(self):
        """Test getting health report"""
        monitor = HealthMonitor('test_node')
        monitor.update_heartbeat()
        
        monitor.register_health_check('test_check', lambda: True)
        monitor.register_component('test_component')
        
        report = monitor.get_health_report()
        
        assert report['node_name'] == 'test_node'
        assert report['overall_status'] == 'healthy'
        assert report['heartbeat_healthy'] is True
        assert 'test_check' in report['health_checks']
        assert 'test_component' in report['components']
        assert 'uptime' in report

