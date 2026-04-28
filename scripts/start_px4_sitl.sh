#!/bin/bash
# Script để khởi động PX4 SITL với Gazebo

set -e

echo "=== KHỞI ĐỘNG PX4 SITL VỚI GAZEBO ==="
echo ""

# Kiểm tra PX4-Autopilot
PX4_DIR="$HOME/PX4-Autopilot"
if [ ! -d "$PX4_DIR" ]; then
    echo "ERROR: PX4-Autopilot không tìm thấy tại $PX4_DIR"
    echo "Vui lòng clone PX4-Autopilot hoặc cập nhật đường dẫn"
    exit 1
fi

cd "$PX4_DIR"

# Kiểm tra và build PX4 nếu cần
if [ ! -f "build/px4_sitl_default/bin/px4" ]; then
    echo "PX4 chưa được build. Đang build..."
    make px4_sitl
fi

# Chọn world và model
# QUAN TRỌNG: Dùng iris_vision vì dự án VUAV-DRONE-WITHOUT-GPS không dùng GPS
# iris_vision tự động set EKF2 cho External Vision + tắt GPS
WORLD=${1:-empty}  # empty, iris, standard_vtol
MODEL=${2:-gazebo-classic_iris_vision}   # gazebo-classic_iris_vision (no GPS), iris (GPS), etc.

echo "World: $WORLD"
echo "Model: $MODEL"
echo ""

# Khởi động PX4 SITL
echo "Khởi động PX4 SITL..."
export PX4_SIM_MODEL=$MODEL
export PX4_SIM_WORLD=$WORLD

# Chạy PX4 SITL với Gazebo (sử dụng gazebo-classic)
./Tools/simulation/gazebo-classic/sitl_multiple_run.sh -n 1 -m $MODEL -w $WORLD

echo ""
echo "PX4 SITL đã khởi động!"
echo "Để dừng, nhấn Ctrl+C"

