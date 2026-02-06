#!/bin/bash
# Script để chạy tests

set -e

echo "=== Running UAV Vision Tests ==="

cd ~/UAV-no-GPS/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

# Run Python tests
python3 -m pytest ros2_ws/src/uav_vision/uav_vision/test_framework.py -v

# Or run unittest
# python3 ros2_ws/src/uav_vision/uav_vision/test_framework.py

echo "=== Tests Complete ==="

