#!/bin/bash
# ============================================
# HƯỚNG DẪN CHẠY UAV TRONG GAZEBO
# ============================================
# Chạy từng terminal theo thứ tự dưới đây
# ============================================

echo "============================================"
echo "HƯỚNG DẪN CHẠY UAV - 6 TERMINALS"
echo "============================================"
echo ""
echo "MỞ 6 TERMINALS RIÊNG BIỆT và chạy từng lệnh:"
echo ""

cat << 'EOF'

╔══════════════════════════════════════════════════════════════╗
║  TERMINAL 1 - PX4 SITL + GAZEBO                             ║
╚══════════════════════════════════════════════════════════════╝

cd ~/PX4-Autopilot
make px4_sitl gazebo

# GIỮ NGUYÊN TERMINAL NÀY


╔══════════════════════════════════════════════════════════════╗
║  TERMINAL 2 - MAVROS                                         ║
╚══════════════════════════════════════════════════════════════╝

cd /home/dkzdragon02/UAV-no-GPS/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch mavros px4.launch \
fcu_url:=udp://:14540@127.0.0.1:14557 \
gcs_url:=udp://@127.0.0.1:14550 \
tgt_system:=1 tgt_component:=1

# GIỮ NGUYÊN TERMINAL NÀY


╔══════════════════════════════════════════════════════════════╗
║  TERMINAL 3 - DUMMY VISION POSE (QUAN TRỌNG!)               ║
╚══════════════════════════════════════════════════════════════╝

cd /home/dkzdragon02/UAV-no-GPS/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run uav_vision dummy_vision_pose

# GIỮ NGUYÊN TERMINAL NÀY (phải chạy liên tục)


╔══════════════════════════════════════════════════════════════╗
║  TERMINAL 4 - SETPOINT VẬN TỐC (0,0,0)                      ║
╚══════════════════════════════════════════════════════════════╝
# QUAN TRỌNG: Chạy lệnh này TRƯỚC khi ARM!

cd /home/dkzdragon02/UAV-no-GPS/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

# CÁCH 1: Dùng file YAML (KHUYẾN NGHỊ)
ros2 topic pub -r 20 /mavros/setpoint_velocity/cmd_vel geometry_msgs/msg/TwistStamped --file setpoint_zero.yaml

# HOẶC CÁCH 2: Dùng lệnh trực tiếp
# ros2 topic pub -r 20 /mavros/setpoint_velocity/cmd_vel geometry_msgs/msg/TwistStamped "{header: {frame_id: 'map'}, twist: {linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}}"

# GIỮ NGUYÊN LỆNH NÀY ĐANG CHẠY (đừng Ctrl+C)
# Đợi 5-10 giây trước khi chuyển sang TERMINAL 5


╔══════════════════════════════════════════════════════════════╗
║  TERMINAL 5 - ARM + OFFBOARD                                ║
╚══════════════════════════════════════════════════════════════╝
# CHỈ CHẠY SAU KHI TERMINAL 4 ĐÃ CHẠY 5-10 GIÂY!

cd /home/dkzdragon02/UAV-no-GPS/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

# ARM
ros2 service call /mavros/cmd/arming mavros_msgs/srv/CommandBool "{value: true}"

# Đợi 2 giây
sleep 2

# OFFBOARD
ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode "{custom_mode: 'OFFBOARD'}"

# Đợi 2 giây
sleep 2

# KIỂM TRA TRẠNG THÁI
ros2 topic echo /mavros/state

# KỲ VỌNG:
# - connected: true
# - armed: true
# - mode: OFFBOARD (KHÔNG phải AUTO.LOITER!)


╔══════════════════════════════════════════════════════════════╗
║  TERMINAL 6 - BAY LÊN                                       ║
╚══════════════════════════════════════════════════════════════╝
# CHỈ CHẠY KHI ĐÃ ARM + OFFBOARD THÀNH CÔNG!
# Dừng TERMINAL 4 (Ctrl+C) trước khi chạy lệnh này

cd /home/dkzdragon02/UAV-no-GPS/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

# CÁCH 1: Dùng file YAML (KHUYẾN NGHỊ)
ros2 topic pub -r 20 /mavros/setpoint_velocity/cmd_vel geometry_msgs/msg/TwistStamped --file setpoint_up.yaml

# HOẶC CÁCH 2: Dùng lệnh trực tiếp
# ros2 topic pub -r 20 /mavros/setpoint_velocity/cmd_vel geometry_msgs/msg/TwistStamped "{header: {frame_id: 'map'}, twist: {linear: {x: 0.0, y: 0.0, z: 1.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}}"

# UAV sẽ bay lên với vận tốc 1 m/s theo trục Z
# Muốn dừng: Ctrl+C, rồi chạy lại với z: 0.0

EOF

echo ""
echo "============================================"
echo "KIỂM TRA"
echo "============================================"
echo ""
echo "Kiểm tra setpoint có đang chạy:"
echo "  ros2 topic hz /mavros/setpoint_velocity/cmd_vel"
echo ""
echo "Kiểm tra vision pose:"
echo "  ros2 topic echo /mavros/vision_pose/pose"
echo ""
echo "Kiểm tra trạng thái:"
echo "  ros2 topic echo /mavros/state"
echo ""

