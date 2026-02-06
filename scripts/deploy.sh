#!/bin/bash
# Deployment script for UAV Vision System

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "=== UAV Vision System Deployment ==="
echo ""

# Configuration
DEPLOY_MODE=${1:-"local"}  # local, docker
ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}

echo "Deployment mode: $DEPLOY_MODE"
echo "ROS Domain ID: $ROS_DOMAIN_ID"
echo ""

# Check dependencies
echo "Checking dependencies..."
if ! command -v ros2 &> /dev/null && [ "$DEPLOY_MODE" != "docker" ]; then
    echo "Error: ROS2 not found. Install ROS2 Humble or use docker mode."
    exit 1
fi

if [ "$DEPLOY_MODE" == "docker" ]; then
    if ! command -v docker &> /dev/null; then
        echo "Error: Docker not found"
        exit 1
    fi
fi

# Create directories
echo "Creating directories..."
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/config"

# Build
if [ "$DEPLOY_MODE" == "docker" ]; then
    echo "Building Docker image..."
    cd "$PROJECT_DIR"
    docker-compose build
    
    echo ""
    echo "=== Deployment Complete ==="
    echo "Start with: docker-compose up"
    echo "Or in background: docker-compose up -d"
else
    echo "Building ROS2 workspace..."
    cd "$PROJECT_DIR/ros2_ws"
    source /opt/ros/humble/setup.bash
    colcon build --packages-select uav_vision
    
    echo ""
    echo "=== Deployment Complete ==="
    echo "Source workspace:"
    echo "  source /opt/ros/humble/setup.bash"
    echo "  source $PROJECT_DIR/ros2_ws/install/setup.bash"
    echo ""
    echo "Launch system:"
    echo "  ros2 launch uav_vision full_autonomy.launch.py"
fi

