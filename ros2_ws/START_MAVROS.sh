#!/bin/bash
# Script để chạy MAVROS

cd /home/dkzdragon02/UAV-no-GPS/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

echo "============================================"
echo "ĐANG KHỞI ĐỘNG MAVROS..."
echo "============================================"
echo ""

ros2 launch mavros px4.launch \
fcu_url:=udp://:14540@127.0.0.1:14557 \
gcs_url:=udp://@127.0.0.1:14550 \
tgt_system:=1 tgt_component:=1

