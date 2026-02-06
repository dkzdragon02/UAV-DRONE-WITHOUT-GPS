# Dockerfile for UAV Vision System
FROM ros:humble

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    python3-numpy \
    python3-yaml \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /workspace/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy workspace
COPY ros2_ws /workspace/ros2_ws

# Build ROS2 workspace
WORKDIR /workspace/ros2_ws
RUN source /opt/ros/humble/setup.bash && \
    colcon build --packages-select uav_vision

# Create log directory
RUN mkdir -p /workspace/logs

# Set environment
ENV ROS_DOMAIN_ID=0
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD ros2 node list || exit 1

# Default command
CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && \
     source /workspace/ros2_ws/install/setup.bash && \
     ros2 launch uav_vision full_autonomy.launch.py"]

