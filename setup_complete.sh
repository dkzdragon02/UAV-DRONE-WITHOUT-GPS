#!/bin/bash
# Complete Setup Script cho UAV Vision System

set -e

echo "=========================================="
echo "UAV Vision System - Complete Setup"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo -e "${RED}Please do not run as root${NC}"
   exit 1
fi

# Check ROS2 installation
echo -e "${YELLOW}Checking ROS2 installation...${NC}"
if [ -z "$ROS_DISTRO" ]; then
    if [ -f /opt/ros/humble/setup.bash ]; then
        echo -e "${GREEN}ROS2 Humble found, sourcing...${NC}"
        source /opt/ros/humble/setup.bash
    else
        echo -e "${RED}ROS2 Humble not found! Please install ROS2 Humble first.${NC}"
        echo "See: https://docs.ros.org/en/humble/Installation.html"
        exit 1
    fi
else
    echo -e "${GREEN}ROS2 $ROS_DISTRO is already sourced${NC}"
fi

# Install system dependencies
echo -e "${YELLOW}Installing system dependencies...${NC}"
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-colcon-common-extensions \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-visualization-msgs \
    ros-humble-diagnostic-msgs \
    ros-humble-nav-msgs \
    ros-humble-geometry-msgs \
    ros-humble-sensor-msgs \
    ros-humble-std-msgs \
    ros-humble-tf2-ros \
    ros-humble-vision-msgs

# Install Python dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
cd "$(dirname "$0")"
pip3 install --user -r requirements.txt

# Build ROS2 workspace
echo -e "${YELLOW}Building ROS2 workspace...${NC}"
if [ -d "ros2_ws" ]; then
    cd ros2_ws
    source /opt/ros/humble/setup.bash
    colcon build --symlink-install
    echo -e "${GREEN}Workspace built successfully!${NC}"
else
    echo -e "${RED}ros2_ws directory not found!${NC}"
    exit 1
fi

# Setup PX4 (optional)
echo -e "${YELLOW}Checking PX4 installation...${NC}"
if [ -d ~/PX4-Autopilot ]; then
    echo -e "${GREEN}PX4-Autopilot found${NC}"
    echo "To setup PX4, run: cd ~/PX4-Autopilot && bash ./Tools/setup/ubuntu.sh"
else
    echo -e "${YELLOW}PX4-Autopilot not found (optional)${NC}"
    echo "To install PX4: git clone https://github.com/PX4/PX4-Autopilot.git ~/PX4-Autopilot"
fi

# Create calibration directories
echo -e "${YELLOW}Creating calibration directories...${NC}"
mkdir -p data/calib_images
mkdir -p data/calib_results
mkdir -p logs

# Make scripts executable
echo -e "${YELLOW}Making scripts executable...${NC}"
[ -d scripts ] && chmod +x scripts/*.sh 2>/dev/null || true
[ -d tools ] && chmod +x tools/*.py 2>/dev/null || true

echo ""
echo -e "${GREEN}=========================================="
echo "Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Source the workspace:"
echo "   cd ~/VUAV-DRONE-WITHOUT-GPS/ros2_ws"
echo "   source /opt/ros/humble/setup.bash"
echo "   source install/setup.bash"
echo ""
echo "2. Calibrate camera (if needed):"
echo "   python3 tools/calibrate_camera_improved.py --images data/calib_images/ --out data/calib_results/camera.yaml"
echo ""
echo "3. Launch the system:"
echo "   ros2 launch uav_vision full_autonomy.launch.py"
echo ""
echo "See COMPLETE_GUIDE.md for detailed instructions"

