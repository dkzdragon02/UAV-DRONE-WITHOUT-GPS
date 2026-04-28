#!/bin/bash
# Script test nhanh với PX4 SITL

echo "=== QUICK TEST PX4 SITL + VISION ==="
echo ""

# Source ROS2
source /opt/ros/humble/setup.bash
cd ~/VUAV-DRONE-WITHOUT-GPS/ros2_ws
source install/setup.bash

echo "Kiểm tra các thành phần:"
echo ""

# 1. Kiểm tra PX4
if [ -d "$HOME/PX4-Autopilot" ]; then
    echo "✅ PX4-Autopilot found"
else
    echo "❌ PX4-Autopilot not found"
fi

# 2. Kiểm tra Gazebo
if command -v gazebo &> /dev/null; then
    echo "✅ Gazebo installed"
else
    echo "❌ Gazebo not installed"
fi

# 3. Kiểm tra pymavlink
if python3 -c "import pymavlink" 2>/dev/null; then
    echo "✅ pymavlink installed"
else
    echo "⚠️  pymavlink not installed (cần: pip3 install pymavlink)"
fi

# 4. Kiểm tra ROS2 workspace
if [ -d "$HOME/VUAV-DRONE-WITHOUT-GPS/ros2_ws/install" ]; then
    echo "✅ ROS2 workspace built"
else
    echo "⚠️  ROS2 workspace not built (cần: cd ros2_ws && colcon build)"
fi

echo ""
echo "=== HƯỚNG DẪN CHẠY (ROS2-ONLY) ==="
echo ""
echo "1. Terminal 1 - PX4 SITL:"
echo "   cd ~/PX4-Autopilot"
echo "   ~/VUAV-DRONE-WITHOUT-GPS/scripts/start_px4_sitl.sh empty iris"
echo ""
echo "2. Terminal 2 - Vision System (ROS2-only):"
echo "   cd ~/UAV-no-GPS/ros2_ws"
echo "   source /opt/ros/humble/setup.bash"
echo "   source install/setup.bash"
echo "   ros2 launch uav_vision px4_ros2_only.launch.py"
echo ""
echo "Hoặc launch toàn bộ:"
echo "   ros2 launch uav_vision px4_sitl_vision.launch.py"
echo ""
echo "3. Terminal 3 - Test Images (optional):"
echo "   ros2 run uav_vision test_image_publisher"
echo ""
echo "=== KIỂM TRA KẾT NỐI ==="
echo ""
echo "# Kiểm tra PX4 bridge connection"
echo "ros2 topic echo /px4_mavlink/connected"
echo ""
echo "# Kiểm tra vision odometry"
echo "ros2 topic echo /uav/vision/odometry"
echo ""
echo "# Kiểm tra status"
echo "ros2 topic echo /px4_mavlink/status"
echo ""

