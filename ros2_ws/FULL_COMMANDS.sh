#!/bin/bash
# ============================================
# HƯỚNG DẪN CHẠY UAV TRONG GAZEBO
# (Vision-based, No GPS)
# ============================================
# Chạy từng terminal theo thứ tự dưới đây
# ============================================

echo "============================================"
echo "HƯỚNG DẪN CHẠY UAV - 6 TERMINALS"
echo "(Vision-based OFFBOARD, No GPS)"
echo "============================================"
echo ""
echo "MỞ 6 TERMINALS RIÊNG BIỆT và chạy từng lệnh:"
echo ""

cat << 'EOF'

╔══════════════════════════════════════════════════════════════╗
║  TERMINAL 1 - PX4 SITL + GAZEBO (iris_vision - KHÔNG GPS)    ║
╚══════════════════════════════════════════════════════════════╝

cd ~/PX4-Autopilot
make px4_sitl gazebo-classic_iris_vision

# QUAN TRỌNG: Phải dùng gazebo-classic_iris_vision (KHÔNG PHẢI iris!)
# iris_vision tự động cấu hình:
#   - EKF2_EV_CTRL = 15 (External Vision)
#   - EKF2_HGT_REF = 3  (Vision height)
#   - EKF2_GPS_CTRL = 0  (Tắt GPS)

# SAU KHI PX4 KHỞI ĐỘNG, paste các lệnh sau vào pxh> console:
#
#   param set COM_LOW_BAT_ACT 0
#   param set NAV_RCL_ACT 0
#   param set COM_RCL_EXCEPT 4
#   param set NAV_DLL_ACT 0
#   param set GF_ACTION 0
#   param set COM_DISARM_PRFLT -1
#   param set COM_OF_LOSS_T 5.0
#   param set BAT_LOW_THR 0.05
#   param set BAT_CRIT_THR 0.03
#   param set BAT_EMERGEN_THR 0.01
#
# Hoặc chạy node tự động (Terminal 3b bên dưới)

# GIỮ NGUYÊN TERMINAL NÀY


╔══════════════════════════════════════════════════════════════╗
║  TERMINAL 2 - MAVROS                                         ║
╚══════════════════════════════════════════════════════════════╝

cd ~/VUAV-DRONE-WITHOUT-GPS/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch mavros px4.launch \
fcu_url:=udp://:14540@127.0.0.1:14557 \
gcs_url:=udp://@127.0.0.1:14550 \
tgt_system:=1 tgt_component:=1

# GIỮ NGUYÊN TERMINAL NÀY


╔══════════════════════════════════════════════════════════════╗
║  TERMINAL 3 - DUMMY VISION POSE (QUAN TRỌNG!)                ║
╚══════════════════════════════════════════════════════════════╝

cd ~/VUAV-DRONE-WITHOUT-GPS/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run uav_vision dummy_vision_pose

# GIỮ NGUYÊN TERMINAL NÀY (phải chạy liên tục)


╔══════════════════════════════════════════════════════════════╗
║  TERMINAL 3b - TỰ ĐỘNG SET PX4 PARAMS (Tùy chọn)             ║
╚══════════════════════════════════════════════════════════════╝
# Thay cho việc paste params thủ công trong Terminal 1
# Node này tự động set tất cả params qua MAVROS

cd ~/VUAV-DRONE-WITHOUT-GPS/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run uav_vision px4_param_setup

# Node sẽ tự tắt sau khi set xong params
# Đợi thấy "PX4 Param Setup complete!" rồi chuyển sang Terminal 4


╔══════════════════════════════════════════════════════════════╗
║  TERMINAL 4 - SETPOINT VẬN TỐC (0,0,0)                       ║
╚══════════════════════════════════════════════════════════════╝
# QUAN TRỌNG: Chạy lệnh này TRƯỚC khi ARM!

cd ~/VUAV-DRONE-WITHOUT-GPS/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 topic pub -r 20 /mavros/setpoint_velocity/cmd_vel geometry_msgs/msg/TwistStamped "{header: {frame_id: 'map'}, twist: {linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}}"

# GIỮ NGUYÊN LỆNH NÀY ĐANG CHẠY (đừng Ctrl+C)
# Đợi 5-10 giây trước khi chuyển sang TERMINAL 5


╔══════════════════════════════════════════════════════════════╗
║  TERMINAL 5 - ARM + OFFBOARD                                 ║
╚══════════════════════════════════════════════════════════════╝
# CHỈ CHẠY SAU KHI TERMINAL 4 ĐÃ CHẠY 5-10 GIÂY!

cd ~/VUAV-DRONE-WITHOUT-GPS/ros2_ws
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
# - mode: OFFBOARD (KHÔNG phải AUTO.LOITER hay AUTO.RTL!)


╔══════════════════════════════════════════════════════════════╗
║  TERMINAL 6 - BAY LÊN                                        ║
╚══════════════════════════════════════════════════════════════╝
# CHỈ CHẠY KHI ĐÃ ARM + OFFBOARD THÀNH CÔNG!
# Dừng TERMINAL 4 (Ctrl+C) trước khi chạy lệnh này

cd ~/VUAV-DRONE-WITHOUT-GPS/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 topic pub -r 20 /mavros/setpoint_velocity/cmd_vel geometry_msgs/msg/TwistStamped "{header: {frame_id: 'map'}, twist: {linear: {x: 0.0, y: 0.0, z: 1.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}}"

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
echo "Kiểm tra PX4 params (trong pxh>):"
echo "  param show EKF2_EV_CTRL     # Phải = 15"
echo "  param show EKF2_GPS_CTRL    # Phải = 0"
echo "  param show COM_LOW_BAT_ACT  # Phải = 0"
echo ""
echo "============================================"
echo "TROUBLESHOOTING"
echo "============================================"
echo ""
echo "Nếu vẫn bị failsafe:"
echo "  1. Kiểm tra dùng đúng model: gazebo-classic_iris_vision"
echo "  2. Kiểm tra dummy_vision_pose đang chạy"
echo "  3. Kiểm tra params đã set đúng (COM_LOW_BAT_ACT = 0)"
echo "  4. Thử xóa params cũ: rm ~/PX4-Autopilot/build/px4_sitl_default/tmp/rootfs/parameters.bson"
echo ""
