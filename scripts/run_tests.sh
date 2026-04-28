#!/bin/bash
# Script để chạy tests

set -e

echo "=== Running UAV Vision Tests ==="

cd ~/VUAV-DRONE-WITHOUT-GPS/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

# Run Python tests
python3 -m pytest src/uav_vision/uav_vision/test_framework.py -v

echo "=== Tests Complete ==="

