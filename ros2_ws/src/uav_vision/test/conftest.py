"""
Pytest configuration and fixtures
"""

import pytest
import sys
import os
from pathlib import Path

# Add source to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

# ROS2 imports require ROS environment
# These will be skipped if ROS2 is not available
try:
    import rclpy
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


@pytest.fixture
def ros2_available():
    """Fixture to check if ROS2 is available"""
    return ROS2_AVAILABLE


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary for testing"""
    return {
        'processing_rate': 30.0,
        'image_topic': '/camera/image_raw',
        'output_topic': '/output/odometry',
        'timeout': 5.0,
        'max_retries': 3
    }


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file"""
    import yaml
    config_file = tmp_path / "test_config.yaml"
    config_data = {
        'test_key': 'test_value',
        'numeric_value': 42,
        'list_value': [1, 2, 3]
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    return str(config_file)

