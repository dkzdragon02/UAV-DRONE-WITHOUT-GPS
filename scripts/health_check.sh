#!/bin/bash
# Health check script for UAV Vision System

set -e

# Configuration
ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
TIMEOUT=10

# Check ROS2 nodes
echo "Checking ROS2 nodes..."
if ! ros2 node list &> /dev/null; then
    echo "ERROR: ROS2 not available"
    exit 1
fi

# Check critical nodes
CRITICAL_NODES=(
    "px4_mavlink_bridge"
    "vision_node"
    "state_machine"
)

ALL_HEALTHY=true

for node in "${CRITICAL_NODES[@]}"; do
    if ros2 node list | grep -q "$node"; then
        echo "✓ $node is running"
    else
        echo "✗ $node is NOT running"
        ALL_HEALTHY=false
    fi
done

# Check topics
echo ""
echo "Checking topics..."
REQUIRED_TOPICS=(
    "/uav/vision/odometry"
    "/px4_mavlink/connected"
)

for topic in "${REQUIRED_TOPICS[@]}"; do
    if ros2 topic list | grep -q "$topic"; then
        echo "✓ $topic exists"
    else
        echo "✗ $topic NOT found"
        ALL_HEALTHY=false
    fi
done

# Check PX4 connection
echo ""
echo "Checking PX4 connection..."
if ros2 topic echo /px4_mavlink/connected --once 2>/dev/null | grep -q "data: true"; then
    echo "✓ PX4 connected"
else
    echo "✗ PX4 NOT connected"
    ALL_HEALTHY=false
fi

# Exit status
if [ "$ALL_HEALTHY" = true ]; then
    echo ""
    echo "=== System Health: OK ==="
    exit 0
else
    echo ""
    echo "=== System Health: DEGRADED ==="
    exit 1
fi

