#!/bin/bash
###############################################################################
# launch_sitl_mavros.sh — 1-Click SITL + MAVROS + Offboard Flight
#
# Mở tmux session với 6 pane, tự động khởi động toàn bộ stack:
#   1. PX4 SITL (iris_vision)
#   2. MAVROS
#   3. Dummy Vision Pose
#   4. PX4 Param Setup
#   5. Setpoint Zero (giữ kết nối)
#   6. ARM + OFFBOARD (sau delay)
#
# Sử dụng:
#   ./scripts/launch_sitl_mavros.sh        → chạy bình thường
#   ./scripts/launch_sitl_mavros.sh --kill  → dừng session
#
# Yêu cầu: tmux, PX4-Autopilot, ROS2 Humble, MAVROS
###############################################################################

set -e

SESSION_NAME="vuav_sitl"
PX4_DIR="${HOME}/PX4-Autopilot"
ROS2_WS="${HOME}/VUAV-DRONE-WITHOUT-GPS/ros2_ws"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ───────────────────────────────── Kill mode ──────────────────────────────── #
if [[ "$1" == "--kill" ]]; then
    echo -e "${RED}Killing session '${SESSION_NAME}'...${NC}"
    tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
    echo -e "${GREEN}Done.${NC}"
    exit 0
fi

# ──────────────────────────────── Preflight ───────────────────────────────── #
if ! command -v tmux &>/dev/null; then
    echo -e "${RED}ERROR: tmux not found. Install with: sudo apt install tmux${NC}"
    exit 1
fi

if [ ! -d "${PX4_DIR}" ]; then
    echo -e "${RED}ERROR: PX4-Autopilot not found at ${PX4_DIR}${NC}"
    exit 1
fi

if [ ! -d "${ROS2_WS}/install" ]; then
    echo -e "${YELLOW}WARNING: ROS2 workspace not built. Building now...${NC}"
    cd "${ROS2_WS}" && source /opt/ros/humble/setup.bash && colcon build --symlink-install
fi

# Kill existing session if running
tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════╗"
echo "║   VUAV — GPS-Denied SITL Flight (MAVROS mode)     ║"
echo "╚═══════════════════════════════════════════════════╝"
echo -e "${NC}"

SOURCE_ROS="source /opt/ros/humble/setup.bash && source ${ROS2_WS}/install/setup.bash"
tmux new-session -d -s "${SESSION_NAME}" -n "main"
tmux split-window -h -t "${SESSION_NAME}:main"
tmux split-window -v -t "${SESSION_NAME}:main.0"
tmux split-window -v -t "${SESSION_NAME}:main.2"
tmux split-window -v -t "${SESSION_NAME}:main.1"
tmux split-window -v -t "${SESSION_NAME}:main.4"

echo -e "${GREEN}[1/6] Starting PX4 SITL (iris_vision)...${NC}"
tmux send-keys -t "${SESSION_NAME}:main.0" \
    "export LIBGL_ALWAYS_SOFTWARE=1 && export GAZEBO_MODEL_DATABASE_URI='' && cd ${PX4_DIR} && make px4_sitl gazebo-classic_iris_vision" Enter

echo -e "${YELLOW}Waiting 30s for PX4 SITL + Gazebo...${NC}"
sleep 30

echo -e "${GREEN}[2/6] Starting MAVROS...${NC}"
tmux send-keys -t "${SESSION_NAME}:main.1" \
    "${SOURCE_ROS} && ros2 launch mavros px4.launch fcu_url:='udp://:14540@127.0.0.1:14557'" Enter

sleep 10

echo -e "${GREEN}[3/6] Starting Dummy Vision Pose...${NC}"
tmux send-keys -t "${SESSION_NAME}:main.2" \
    "${SOURCE_ROS} && ros2 run uav_vision dummy_vision_pose" Enter

sleep 5

echo -e "${GREEN}[4/6] Setting PX4 parameters...${NC}"
tmux send-keys -t "${SESSION_NAME}:main.3" \
    "${SOURCE_ROS} && ros2 run uav_vision px4_param_setup" Enter

sleep 15

echo -e "${GREEN}[5/6] Starting setpoint stream (zero velocity)...${NC}"
tmux send-keys -t "${SESSION_NAME}:main.4" \
    "${SOURCE_ROS} && ros2 topic pub -r 20 /mavros/setpoint_velocity/cmd_vel geometry_msgs/msg/TwistStamped \"{header: {frame_id: 'map'}, twist: {linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}}\"" Enter

sleep 10

echo -e "${GREEN}[6/6] ARM + OFFBOARD mode...${NC}"
tmux send-keys -t "${SESSION_NAME}:main.5" \
    "${SOURCE_ROS} && echo '>>> ARMing...' && ros2 service call /mavros/cmd/arming mavros_msgs/srv/CommandBool \"{value: true}\" && sleep 3 && echo '>>> Setting OFFBOARD mode...' && ros2 service call /mavros/set_mode mavros_msgs/srv/SetMode \"{custom_mode: 'OFFBOARD'}\" && echo '>>> DONE! Drone is armed and in OFFBOARD mode.'" Enter

echo ""
echo -e "${CYAN}╔═══════════════════════════════════════════════════╗"
echo "║  All 6 panes started!                                       ║"
echo "║                                                             ║"
echo "║  Attach:  tmux attach -t ${SESSION_NAME}                    ║"
echo "║  Kill:    $0 --kill                                         ║"
echo "║                                                             ║"
echo "║  To fly up (in any terminal):                               ║"
echo "║  ros2 topic pub -r 20 \\                                    ║"
echo "║    /mavros/setpoint_velocity/cmd_vel \\                     ║"
echo "║    geometry_msgs/msg/TwistStamped \\                        ║"
echo "║    \"{..., z: 1.0, ...}\"                                   ║"
echo "╚═════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

tmux attach -t "${SESSION_NAME}"
