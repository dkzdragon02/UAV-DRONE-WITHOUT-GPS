# Hệ thống UAV Không GPS – PX4 SITL + ROS2 MAVROS

Tài liệu này hướng dẫn chạy mô phỏng UAV không GPS với PX4 SITL (Gazebo Classic) và ROS2 Humble + MAVROS, bao gồm luồng OFFBOARD và cấp dữ liệu vision giả lập cho EKF.

---

## Nội dung chính
- Mục tiêu & kiến trúc
- Yêu cầu hệ thống
- Cài đặt & build ROS2 workspace
- Quy trình chạy mô phỏng 6 terminal (PX4 + MAVROS + vision + setpoint)
- Cấu trúc thư mục
- Thành phần ROS2 quan trọng
- Kiểm tra & troubleshooting nhanh
- An toàn & lưu ý

---

## Mục tiêu
- Bay OFFBOARD trong mô phỏng Gazebo Classic mà không cần GPS.
- Cung cấp vision pose giả (`dummy_vision_pose`) cho PX4 EKF để vượt qua lỗi `ekf2 missing data`.
- Stream setpoint velocity liên tục để giữ OFFBOARD và arming ổn định.

---

## Yêu cầu hệ thống
- Ubuntu 22.04 + ROS2 Humble + colcon.
- PX4-Autopilot đã clone tại `~/PX4-Autopilot` (sử dụng gazebo-classic).
- MAVROS cho ROS2 Humble.
- Python3, pip, và các package trong `requirements.txt` (nếu dùng stack ngoài ROS2).
- Dung lượng đĩa đủ cho build PX4 và workspace.

---

## Cài đặt & Build ROS2 workspace
```bash
cd /home/dkzdragon02/UAV-no-GPS/ros2_ws
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

> Mỗi terminal ROS2 phải `source /opt/ros/humble/setup.bash` rồi `source install/setup.bash`.

---

## Quy trình chạy mô phỏng (6 terminal)
File tóm tắt lệnh: `ros2_ws/FULL_COMMANDS.sh` (echo hướng dẫn).

**Terminal 1 – PX4 SITL + Gazebo**
```bash
cd ~/PX4-Autopilot
make px4_sitl gazebo   # hoặc dùng scripts/start_px4_sitl.sh empty iris
```

**Terminal 2 – MAVROS**
```bash
cd /home/dkzdragon02/UAV-no-GPS/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch mavros px4.launch \
  fcu_url:=udp://:14540@127.0.0.1:14557 \
  gcs_url:=udp://@127.0.0.1:14550 \
  tgt_system:=1 tgt_component:=1
```

**Terminal 3 – Dummy Vision Pose (quan trọng)**
```bash
cd /home/dkzdragon02/UAV-no-GPS/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run uav_vision dummy_vision_pose
```

**Terminal 4 – Stream setpoint vận tốc (0,0,0) trước khi ARM**
```bash
cd /home/dkzdragon02/UAV-no-GPS/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic pub -r 20 /mavros/setpoint_velocity/cmd_vel geometry_msgs/msg/TwistStamped --file setpoint_zero.yaml
```
Giữ lệnh này chạy liên tục ≥5–10 s trước khi ARM.

**Terminal 5 – ARM + OFFBOARD**
```bash
cd /home/dkzdragon02/UAV-no-GPS/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 service call /mavros/cmd/arming mavros_msgs/srv/CommandBool "{value: true}"
sleep 2
ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode "{custom_mode: 'OFFBOARD'}"
sleep 2
ros2 topic echo /mavros/state
# Kỳ vọng: connected=true, armed=true, mode=OFFBOARD
```

**Terminal 6 – Bay lên (z=1 m/s)**
```bash
# Dừng Terminal 4 (Ctrl+C) trước khi đổi setpoint
cd /home/dkzdragon02/UAV-no-GPS/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic pub -r 20 /mavros/setpoint_velocity/cmd_vel geometry_msgs/msg/TwistStamped --file setpoint_up.yaml
```
Muốn dừng: Ctrl+C rồi gửi lại `setpoint_zero.yaml`.

---

## Kiểm tra nhanh
- Tốc độ setpoint: `ros2 topic hz /mavros/setpoint_velocity/cmd_vel`
- Vision pose: `ros2 topic echo /mavros/vision_pose/pose`
- Trạng thái PX4: `ros2 topic echo /mavros/state`
- Nếu ARM bị từ chối trong PX4 shell: gõ `commander arm` để thấy lý do.

---

## Cấu trúc thư mục
```
UAV-no-GPS/
├── README.md
├── requirements.txt
├── ros2_ws/
│   ├── FULL_COMMANDS.sh        # Hướng dẫn lệnh nhanh
│   ├── START_MAVROS.sh         # Chạy MAVROS
│   ├── setpoint_zero.yaml      # Setpoint vận tốc (0,0,0)
│   ├── setpoint_up.yaml        # Setpoint bay lên (0,0,1)
│   └── src/uav_vision/...
├── scripts/
│   └── start_px4_sitl.sh       # Khởi động PX4 SITL + Gazebo
└── tools/ ...                  # Tiện ích bổ trợ
```

---

## Thành phần ROS2 quan trọng (`ros2_ws/src/uav_vision`)
- `uav_vision/dummy_vision_pose.py`: xuất `PoseStamped` đến `/mavros/vision_pose/pose` và `Odometry` đến `/uav/vision/odometry` (cần cho EKF).
- `uav_vision/px4_mavros_offboard_bridge.py`: bridge vận tốc → MAVROS, auto ARM/OFFBOARD (cần MAVROS sẵn sàng).
- `uav_vision/px4_controller.py`: tạo setpoint velocity từ state machine/path.
- `uav_vision/state_machine.py`: quản lý trạng thái bay (IDLE, TAKEOFF, HOVER, LAND, ...).
- `launch/full_autonomy.launch.py`: launch toàn bộ stack (VO/SLAM, state machine, controller, bridge).

---

## Troubleshooting nhanh
- `armed` vẫn false dù gọi ARM:
  - Đảm bảo Terminal 4 đã stream setpoint ≥20 Hz trước ARM.
  - Đảm bảo Terminal 3 (dummy_vision_pose) đang chạy, PX4 không báo `ekf2 missing data`.
  - Kiểm tra `/mavros/state` có `connected: true`.
  - Trong PX4 shell, chạy `commander arm` để xem lỗi chi tiết.
- MAVROS báo “Waiting for /mavros/state”:
  - Kiểm tra đã `source /opt/ros/humble/setup.bash && source install/setup.bash` trong terminal MAVROS.
- Lỗi YAML `ros2 topic pub`:
  - Dùng `--file setpoint_zero.yaml` hoặc `setpoint_up.yaml` để tránh sai cú pháp.

---

## An toàn & lưu ý
- Chỉ thử bay thật khi đã hiểu quy trình OFFBOARD và có khu vực an toàn.
- Luôn giữ một terminal PX4 shell để có thể `commander disarm` nhanh.
- Trong mô phỏng, giữ setpoint liên tục; OFFBOARD sẽ rơi về chế độ khác nếu mất setpoint.
- Nếu đã chỉnh tham số PX4 thử nghiệm và gặp lỗi lạ, có thể xoá `build/px4_sitl_default/rootfs` để reset SITL.

---

## Giấy phép
MIT License

