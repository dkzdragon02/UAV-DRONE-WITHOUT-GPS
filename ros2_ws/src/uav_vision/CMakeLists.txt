cmake_minimum_required(VERSION 3.8)
project(uav_vision)

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

# Manually install entry point scripts since ament_python_install_package
# generates its own setup.py without entry_points
install(PROGRAMS
  scripts/vision_node
  scripts/mavros_bridge
  scripts/optical_flow_node
  scripts/test_image_publisher
  scripts/px4_mavlink_bridge
  scripts/obstacle_avoidance_node
  scripts/mock_lidar_publisher
  scripts/slam_node
  scripts/state_machine
  scripts/path_planner
  scripts/px4_controller
  scripts/coverage_planner_node
  scripts/evaluation_node
  scripts/performance_monitor
  scripts/visualization_node
  scripts/px4_mavros_offboard_bridge
  scripts/dummy_vision_pose
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch resource config
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/dummy.sh
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

