#!/usr/bin/env python3
"""
Health Monitor and Watchdog System
Monitors node health and provides watchdog functionality
"""

import time
import threading
from typing import Dict, Callable, Optional, List
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
import logging


class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Health check definition"""
    name: str
    check_func: Callable[[], bool]
    timeout: float = 5.0
    critical: bool = False
    last_check: float = field(default_factory=time.time)
    last_result: bool = True
    consecutive_failures: int = 0
    max_failures: int = 3


@dataclass
class ComponentHealth:
    """Component health information"""
    name: str
    status: HealthStatus
    last_update: float
    uptime: float
    error_count: int
    checks: Dict[str, bool] = field(default_factory=dict)
    message: str = ""


class HealthMonitor:
    """
    Health monitor and watchdog system.
    
    Monitors:
    - Node heartbeat
    - Component health
    - Resource usage
    - Error rates
    """
    
    def __init__(
        self,
        node_name: str,
        heartbeat_timeout: float = 5.0,
        check_interval: float = 1.0
    ):
        """
        Initialize health monitor.
        
        Args:
            node_name: Name of the node being monitored
            heartbeat_timeout: Heartbeat timeout in seconds
            check_interval: Health check interval in seconds
        """
        self.node_name = node_name
        self.heartbeat_timeout = heartbeat_timeout
        self.check_interval = check_interval
        
        self.logger = logging.getLogger(__name__)
        
        # Health checks
        self._health_checks: Dict[str, HealthCheck] = {}
        self._components: Dict[str, ComponentHealth] = {}
        
        # Heartbeat tracking
        self._last_heartbeat = time.time()
        self._heartbeat_lock = threading.Lock()
        
        # Monitoring thread
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_lock = threading.Lock()
        
        # Statistics
        self._error_history = deque(maxlen=100)
        self._uptime_start = time.time()
    
    def register_health_check(
        self,
        name: str,
        check_func: Callable[[], bool],
        timeout: float = 5.0,
        critical: bool = False,
        max_failures: int = 3
    ):
        """
        Register a health check.
        
        Args:
            name: Name of the health check
            check_func: Function that returns True if healthy
            timeout: Timeout for the check
            critical: Whether this is a critical check
            max_failures: Maximum consecutive failures before unhealthy
        """
        check = HealthCheck(
            name=name,
            check_func=check_func,
            timeout=timeout,
            critical=critical,
            max_failures=max_failures
        )
        self._health_checks[name] = check
        self.logger.info(f"Registered health check: {name}")
    
    def register_component(self, name: str):
        """
        Register a component for monitoring.
        
        Args:
            name: Component name
        """
        component = ComponentHealth(
            name=name,
            status=HealthStatus.HEALTHY,
            last_update=time.time(),
            uptime=0.0,
            error_count=0
        )
        self._components[name] = component
        self.logger.info(f"Registered component: {name}")
    
    def update_component_health(
        self,
        name: str,
        status: HealthStatus,
        error_count: int = 0,
        message: str = ""
    ):
        """
        Update component health status.
        
        Args:
            name: Component name
            status: Health status
            error_count: Error count
            message: Status message
        """
        if name not in self._components:
            self.register_component(name)
        
        component = self._components[name]
        component.status = status
        component.last_update = time.time()
        component.error_count = error_count
        component.message = message
        component.uptime = time.time() - self._uptime_start
    
    def update_heartbeat(self):
        """Update heartbeat timestamp."""
        with self._heartbeat_lock:
            self._last_heartbeat = time.time()
    
    def is_heartbeat_healthy(self) -> bool:
        """
        Check if heartbeat is healthy.
        
        Returns:
            True if heartbeat is within timeout, False otherwise
        """
        with self._heartbeat_lock:
            elapsed = time.time() - self._last_heartbeat
            return elapsed < self.heartbeat_timeout
    
    def get_overall_health(self) -> HealthStatus:
        """
        Get overall health status.
        
        Returns:
            Overall health status
        """
        # Check heartbeat
        if not self.is_heartbeat_healthy():
            return HealthStatus.CRITICAL
        
        # Check critical health checks
        for check in self._health_checks.values():
            if check.critical and not check.last_result:
                if check.consecutive_failures >= check.max_failures:
                    return HealthStatus.CRITICAL
        
        # Check components
        critical_components = [
            comp for comp in self._components.values()
            if comp.status == HealthStatus.CRITICAL
        ]
        if critical_components:
            return HealthStatus.CRITICAL
        
        unhealthy_components = [
            comp for comp in self._components.values()
            if comp.status == HealthStatus.UNHEALTHY
        ]
        if unhealthy_components:
            return HealthStatus.UNHEALTHY
        
        degraded_components = [
            comp for comp in self._components.values()
            if comp.status == HealthStatus.DEGRADED
        ]
        if degraded_components:
            return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY
    
    def run_health_checks(self):
        """Run all registered health checks."""
        for name, check in self._health_checks.items():
            try:
                start_time = time.time()
                result = check.check_func()
                elapsed = time.time() - start_time
                
                if elapsed > check.timeout:
                    self.logger.warning(
                        f"Health check '{name}' exceeded timeout: {elapsed:.2f}s > {check.timeout}s"
                    )
                    result = False
                
                check.last_check = time.time()
                
                if result:
                    check.consecutive_failures = 0
                    check.last_result = True
                else:
                    check.consecutive_failures += 1
                    check.last_result = False
                    
                    if check.consecutive_failures >= check.max_failures:
                        self.logger.error(
                            f"Health check '{name}' failed {check.consecutive_failures} times"
                        )
            
            except Exception as e:
                self.logger.error(
                    f"Error running health check '{name}': {e}",
                    exc_info=True
                )
                check.last_result = False
                check.consecutive_failures += 1
    
    def start_monitoring(self):
        """Start health monitoring thread."""
        with self._monitor_lock:
            if self._monitoring:
                return
            
            self._monitoring = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True
            )
            self._monitor_thread.start()
            self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring thread."""
        with self._monitor_lock:
            if not self._monitoring:
                return
            
            self._monitoring = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=2.0)
            self.logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                self.run_health_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}", exc_info=True)
                time.sleep(self.check_interval)
    
    def record_error(self, error: Exception):
        """
        Record an error.
        
        Args:
            error: The exception that occurred
        """
        self._error_history.append({
            "timestamp": time.time(),
            "error": str(error),
            "type": type(error).__name__
        })
    
    def get_health_report(self) -> Dict:
        """
        Get comprehensive health report.
        
        Returns:
            Dictionary with health information
        """
        return {
            "node_name": self.node_name,
            "overall_status": self.get_overall_health().value,
            "heartbeat_healthy": self.is_heartbeat_healthy(),
            "uptime": time.time() - self._uptime_start,
            "health_checks": {
                name: {
                    "status": "healthy" if check.last_result else "unhealthy",
                    "consecutive_failures": check.consecutive_failures,
                    "last_check": check.last_check,
                    "critical": check.critical
                }
                for name, check in self._health_checks.items()
            },
            "components": {
                name: {
                    "status": comp.status.value,
                    "error_count": comp.error_count,
                    "uptime": comp.uptime,
                    "message": comp.message
                }
                for name, comp in self._components.items()
            },
            "recent_errors": list(self._error_history)[-10:]  # Last 10 errors
        }

