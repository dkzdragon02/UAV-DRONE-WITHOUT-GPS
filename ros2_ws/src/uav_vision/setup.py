from setuptools import setup
import os
from glob import glob

package_name = 'uav_vision'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='UAV Team',
    maintainer_email='user@example.com',
    description='UAV Vision Processing Node for GPS-denied navigation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_node = uav_vision.vision_node:main',
            'optical_flow_node = uav_vision.optical_flow_node:main',
            'object_detection_node = uav_vision.object_detection_node:main',
            'slam_node = uav_vision.slam_node:main',
            'state_machine = uav_vision.state_machine:main',
            'path_planner = uav_vision.path_planner:main',
            'px4_controller = uav_vision.px4_controller:main',
            'coverage_planner_node = uav_vision.coverage_planner_node:main',
            'evaluation_node = uav_vision.evaluation_node:main',
            'performance_monitor = uav_vision.performance_monitor:main',
            'visualization_node = uav_vision.visualization_node:main',
            'mavros_bridge = uav_vision.mavros_bridge:main',
            'px4_mavlink_bridge = uav_vision.px4_mavlink_bridge:main',
            'test_image_publisher = uav_vision.test_image_publisher:main',
            'obstacle_avoidance_node = uav_vision.obstacle_avoidance_node:main',
            'mock_lidar_publisher = uav_vision.mock_lidar_publisher:main',
            'px4_mavros_offboard_bridge = uav_vision.px4_mavros_offboard_bridge:main',
            'dummy_vision_pose = uav_vision.dummy_vision_pose:main',
        ],
    },
)

