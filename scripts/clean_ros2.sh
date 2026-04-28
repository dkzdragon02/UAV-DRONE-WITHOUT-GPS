#!/bin/bash
# Clean ROS2 build artifacts

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "=== Cleaning ROS2 Build Artifacts ==="
echo ""

cd "$PROJECT_DIR/ros2_ws"

read -p "This will delete build/, install/, and log/ directories. Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo "Removing build/ directory..."
rm -rf build/

echo "Removing install/ directory..."
rm -rf install/

echo "Removing log/ directory..."
rm -rf log/

echo ""
echo "=== Cleanup Complete ==="
echo "Rebuild with: cd ros2_ws && source /opt/ros/humble/setup.bash && colcon build"

