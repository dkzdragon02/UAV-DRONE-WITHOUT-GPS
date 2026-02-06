# UAV Vision ROS2 Package

# Base classes and utilities
from .base_node import BaseNode, NodeState
from .config_manager import ConfigManager, ConfigSource, ConfigValidator
from .error_handler import ErrorHandler, RetryStrategy, retry_on_error, safe_execute
from .health_monitor import HealthMonitor, HealthStatus

# Utilities
from .utils import (
    Timer,
    quaternion_to_euler,
    euler_to_quaternion,
    normalize_angle,
    distance_2d,
    distance_3d,
    clamp_value,
    wrap_value,
    rate_limiter,
    moving_average,
    exponential_smoothing,
)

# Performance tools
from .profiler import (
    Profiler,
    FunctionProfiler,
    profile_function,
    PerformanceBenchmark,
)

from .memory_profiler import (
    MemoryProfiler,
    MemoryMonitor,
    get_memory_usage,
)

from .performance_optimizer import (
    PerformanceOptimizer,
    AsyncProcessor,
    RateLimiter,
    Cache,
)

# Plugin system
from .plugin_system import (
    PluginInterface,
    SLAMBackendInterface,
    PathPlannerInterface,
    PluginLoader,
    PluginManager,
)

# Advanced planning
from .rrt_star_planner import RRTStarPlanner
from .trajectory_optimizer import TrajectoryOptimizer
from .mission_planner import (
    MissionPlanner,
    Waypoint,
    Mission,
    MissionState,
)

# Logging & Monitoring
from .logger import (
    LoggerManager,
    get_logger,
    PerformanceLogger,
)

from .metrics import (
    MetricsCollector,
    get_metrics_collector,
    MetricType,
)

# Security
from .security import (
    InputValidator,
    Authenticator,
    require_authentication,
    validate_input,
    RateLimiter as SecurityRateLimiter,
)

__all__ = [
    # Base classes
    'BaseNode',
    'NodeState',
    # Configuration
    'ConfigManager',
    'ConfigSource',
    'ConfigValidator',
    # Error handling
    'ErrorHandler',
    'RetryStrategy',
    'retry_on_error',
    'safe_execute',
    # Health monitoring
    'HealthMonitor',
    'HealthStatus',
    # Utilities
    'Timer',
    'quaternion_to_euler',
    'euler_to_quaternion',
    'normalize_angle',
    'distance_2d',
    'distance_3d',
    'clamp_value',
    'wrap_value',
    'rate_limiter',
    'moving_average',
    'exponential_smoothing',
    # Performance tools
    'Profiler',
    'FunctionProfiler',
    'profile_function',
    'PerformanceBenchmark',
    'MemoryProfiler',
    'MemoryMonitor',
    'get_memory_usage',
    'PerformanceOptimizer',
    'AsyncProcessor',
    'RateLimiter',
    'Cache',
    # Plugin system
    'PluginInterface',
    'SLAMBackendInterface',
    'PathPlannerInterface',
    'PluginLoader',
    'PluginManager',
    # Advanced planning
    'RRTStarPlanner',
    'TrajectoryOptimizer',
    'MissionPlanner',
    'Waypoint',
    'Mission',
    'MissionState',
    # Logging & Monitoring
    'LoggerManager',
    'get_logger',
    'PerformanceLogger',
    'MetricsCollector',
    'get_metrics_collector',
    'MetricType',
    # Security
    'InputValidator',
    'Authenticator',
    'require_authentication',
    'validate_input',
    'RateLimiter',
]

