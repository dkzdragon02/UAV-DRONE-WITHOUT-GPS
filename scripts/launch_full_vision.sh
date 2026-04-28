#!/bin/bash
###############################################################################
# launch_full_vision.sh — 1-Click PX4 SITL + MAVROS + Full Vision Stack
#
# Mở tmux session với 4 pane, khởi động:
#   Pane 0: PX4 SITL (iris — proven working target)
#   Pane 1: MAVROS (ros2 run — bypasses XML arg parsing issues)
#   Pane 2: Test Image Publisher (provides /camera/image_raw)
#   Pane 3: Full Vision Autonomy Launch
#
# Sử dụng:
#   ./scripts/launch_full_vision.sh             → chạy đầy đủ
#   ./scripts/launch_full_vision.sh --no-slam   → tắt SLAM
#   ./scripts/launch_full_vision.sh --kill      → dừng session
#
# Yêu cầu: tmux, PX4-Autopilot, ROS2 Humble, MAVROS
###############################################################################

set -euo pipefail

SESSION_NAME="vuav_vision"
PX4_DIR="${HOME}/PX4-Autopilot"
ROS2_WS="${HOME}/VUAV-DRONE-WITHOUT-GPS/ros2_ws"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'
ENABLE_SLAM="true"
ENABLE_VISION="true"
ENABLE_MONITORING="true"
AUTO_TAKEOFF="true"

for arg in "$@"; do
    case $arg in
        --kill)
            echo -e "${RED}Killing session '${SESSION_NAME}'...${NC}"
            tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
            pkill -f "px4_sitl" 2>/dev/null || true
            pkill -f "gzserver" 2>/dev/null || true
            pkill -f "gzclient" 2>/dev/null || true
            echo -e "${GREEN}Done.${NC}"
            exit 0
            ;;
        --no-slam)
            ENABLE_SLAM="false"
            ;;
        --no-vision)
            ENABLE_VISION="false"
            ;;
        --no-monitoring)
            ENABLE_MONITORING="false"
            ;;
        --no-auto-takeoff)
            AUTO_TAKEOFF="false"
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --kill              Kill running session"
            echo "  --no-slam           Disable SLAM node"
            echo "  --no-vision         Disable Vision VO (use dummy pose)"
            echo "  --no-monitoring     Disable performance monitoring"
            echo "  --no-auto-takeoff   Require manual takeoff command"
            echo "  --help              Show this help"
            exit 0
            ;;
    esac
done

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

###############################################################################
# Kill stale sessions & processes
###############################################################################
tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
pkill -f "px4 " 2>/dev/null || true
pkill -f "gzserver" 2>/dev/null || true
pkill -f "gzclient" 2>/dev/null || true
sleep 2

###############################################################################
# Configuration — USE PROVEN WORKING TARGETS ONLY
###############################################################################
# "iris" is the ONLY PX4 SITL target confirmed to work via ninja build.
# iris_vision_camera target fails with "ninja: error: unknown target" because
# cmake hasn't been reconfigured. Using iris + test_image_publisher instead.
PX4_MODEL="iris"

echo -e "${MAGENTA}"
echo "╔═══════════════════════════════════════════════════════╗"
echo "║   VUAV — Full Vision Autonomy (SITL Mode)            ║"
echo "╠═══════════════════════════════════════════════════════╣"
echo "║  PX4 Model: ${PX4_MODEL} (+ test camera publisher)   ║"
echo "║  Vision:    ${ENABLE_VISION}                          ║"
echo "║  SLAM:      ${ENABLE_SLAM}                            ║"
echo "║  Monitor:   ${ENABLE_MONITORING}                      ║"
echo "║  AutoTkoff: ${AUTO_TAKEOFF}                           ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo -e "${NC}"

###############################################################################
# ROS2 source command
###############################################################################
SOURCE_ROS="source /opt/ros/humble/setup.bash && source ${ROS2_WS}/install/setup.bash"

###############################################################################
# PX4 SITL command (proven working target)
###############################################################################
PX4_CMD="export LIBGL_ALWAYS_SOFTWARE=1 && export GAZEBO_MODEL_DATABASE_URI='' && cd ${PX4_DIR} && make px4_sitl gazebo-classic_${PX4_MODEL}"

###############################################################################
# MAVROS command — use ros2 run directly (NOT ros2 launch node.launch)
# The XML launch file fails silently through tmux due to arg parsing issues
###############################################################################
MAVROS_PLUGINLISTS="${ROS2_WS}/install/uav_vision/share/uav_vision/config/mavros_pluginlists.yaml"
MAVROS_CONFIG="/opt/ros/humble/share/mavros/launch/px4_config.yaml"
MAVROS_CMD="${SOURCE_ROS} && ros2 run mavros mavros_node --ros-args -r __ns:=/mavros -p fcu_url:=udp://:14540@127.0.0.1:14557 -p tgt_system:=1 -p tgt_component:=1 -p fcu_protocol:=v2.0 --params-file ${MAVROS_PLUGINLISTS} --params-file ${MAVROS_CONFIG}"

###############################################################################
# Create tmux session with 4 panes (2x2 grid)
###############################################################################
if ! tmux new-session -d -s "${SESSION_NAME}" -n "main" 2>/dev/null; then
    echo -e "${RED}ERROR: Failed to create tmux session '${SESSION_NAME}'.${NC}"
    echo -e "${YELLOW}Try: tmux kill-server && retry${NC}"
    exit 1
fi
tmux split-window -h -t "${SESSION_NAME}:main"
tmux split-window -v -t "${SESSION_NAME}:main.0"
tmux split-window -v -t "${SESSION_NAME}:main.2"

###############################################################################
# Step 1: PX4 SITL (Pane 0 — top-left)
###############################################################################
echo -e "${GREEN}[1/4] Starting PX4 SITL (${PX4_MODEL})...${NC}"
tmux send-keys -t "${SESSION_NAME}:main.0" "${PX4_CMD}" Enter

echo -e "${YELLOW}Waiting 30s for PX4 SITL + Gazebo...${NC}"
sleep 30

###############################################################################
# Step 2: MAVROS (Pane 1 — bottom-left)
###############################################################################
echo -e "${GREEN}[2/4] Starting MAVROS (ros2 run)...${NC}"
tmux send-keys -t "${SESSION_NAME}:main.1" "${MAVROS_CMD}" Enter

sleep 5

###############################################################################
# Step 3: Test Image Publisher (Pane 2 — top-right)
# Provides /camera/image_raw since basic iris model has no camera sensor
###############################################################################
echo -e "${GREEN}[3/4] Starting Test Image Publisher (camera simulator)...${NC}"
CAMERA_CMD="${SOURCE_ROS} && ros2 run uav_vision test_image_publisher --ros-args -p width:=640 -p height:=480 -p fps:=30.0 -p pattern:=terrain"
tmux send-keys -t "${SESSION_NAME}:main.2" "${CAMERA_CMD}" Enter

sleep 3

###############################################################################
# Step 4: Full Vision Autonomy Stack (Pane 3 — bottom-right)
###############################################################################
echo -e "${GREEN}[4/4] Starting Full Vision Autonomy Stack...${NC}"
LAUNCH_CMD="${SOURCE_ROS} && ros2 launch uav_vision full_vision_autonomy.launch.py"
LAUNCH_CMD="${LAUNCH_CMD} auto_takeoff:=${AUTO_TAKEOFF}"
LAUNCH_CMD="${LAUNCH_CMD} enable_vision:=${ENABLE_VISION}"
LAUNCH_CMD="${LAUNCH_CMD} enable_slam:=${ENABLE_SLAM}"
LAUNCH_CMD="${LAUNCH_CMD} enable_monitoring:=${ENABLE_MONITORING}"

tmux send-keys -t "${SESSION_NAME}:main.3" "${LAUNCH_CMD}" Enter

echo ""
echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════════════╗"
echo "║  Full Vision Stack started!                                        ║"
echo "║                                                                    ║"
echo "║  Panes:                                                            ║"
echo "║    0 (top-left)    : PX4 SITL                                      ║"
echo "║    1 (bottom-left) : MAVROS                                        ║"
echo "║    2 (top-right)   : Camera Simulator                              ║"
echo "║    3 (bottom-right): Vision Autonomy Stack                         ║"
echo "║                                                                    ║"
echo "║  Attach:  tmux attach -t ${SESSION_NAME}                           ║"
echo "║  Kill:    $0 --kill                                                ║"
echo "║                                                                    ║"
echo "║  Monitoring commands:                                              ║"
echo "║    ros2 topic echo /uav/state_machine/status                       ║"
echo "║    ros2 topic echo /uav/px4_controller/altitude                    ║"
echo "║    ros2 topic echo /uav/vision/odometry                            ║"
echo "║    ros2 topic hz /camera/image_raw                                 ║"
echo "║    ros2 topic hz /mavros/state                                     ║"
echo "║                                                                    ║"
echo "║  Manual commands:                                                  ║"
echo "║    Takeoff: ros2 topic pub --once \\                               ║"
echo "║      /uav/state_machine/command \\                                 ║"
echo "║      std_msgs/msg/String \"{data: 'takeoff'}\"                     ║"
echo "║    Land:    ...  \"{data: 'land'}\"                                ║"
echo "║    RTL:     ...  \"{data: 'rtl'}\"                                 ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

tmux attach -t "${SESSION_NAME}"
