#!/usr/bin/env python3
"""
ROS2 State Machine Node cho Autonomous Flight
Quản lý các trạng thái bay tự động
"""

import rclpy
from rclpy.node import Node
from enum import Enum
from std_msgs.msg import String, Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import time


class FlightState(Enum):
    """Các trạng thái bay"""
    IDLE = "idle"
    TAKEOFF = "takeoff"
    NAVIGATION = "navigation"
    LANDING = "landing"
    EMERGENCY = "emergency"
    RETURN_TO_LAUNCH = "rtl"
    HOVER = "hover"


class StateMachine(Node):
    """State Machine cho autonomous flight"""
    
    def __init__(self):
        super().__init__('state_machine')
        
        # Current state
        self.current_state = FlightState.IDLE
        self.state_start_time = time.time()
        
        # Parameters
        self.declare_parameter('odom_topic', '/uav/vision/odometry')
        self.declare_parameter('target_altitude', 2.0)  # meters
        self.declare_parameter('takeoff_timeout', 30.0)  # seconds
        self.declare_parameter('landing_timeout', 30.0)  # seconds
        self.declare_parameter('emergency_landing_altitude', 0.5)  # meters
        self.declare_parameter('odom_timeout', 2.0)  # seconds
        
        odom_topic = self.get_parameter('odom_topic').value
        self.target_altitude = self.get_parameter('target_altitude').value
        self.takeoff_timeout = self.get_parameter('takeoff_timeout').value
        self.landing_timeout = self.get_parameter('landing_timeout').value
        self.emergency_landing_altitude = self.get_parameter('emergency_landing_altitude').value
        self.odom_timeout = self.get_parameter('odom_timeout').value
        
        # Current pose
        self.current_pose = None
        self.current_altitude = 0.0
        self.last_odom_time = None
        
        # Mission parameters
        self.waypoints = []
        self.current_waypoint_index = 0
        self.waypoint_tolerance = 0.5  # meters
        
        # Safety checks
        self.max_altitude = 10.0  # meters
        self.min_battery_voltage = 10.5  # volts (example)
        self.vision_ok = False
        self.px4_connected = False
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            10
        )
        
        # Publishers
        self.state_pub = self.create_publisher(String, '/uav/state_machine/state', 10)
        self.command_pub = self.create_publisher(String, '/uav/state_machine/command', 10)
        self.safety_pub = self.create_publisher(Bool, '/uav/state_machine/safety_ok', 10)
        
        # Service clients (for PX4 commands)
        # Có thể thêm service clients để gọi PX4 commands
        
        # Timer for state machine logic
        self.state_timer = self.create_timer(0.1, self.state_machine_logic)
        
        # Command subscriber
        self.command_sub = self.create_subscription(
            String,
            '/uav/state_machine/command',
            self.command_callback,
            10
        )
        
        self.get_logger().info('State Machine started')
        self.get_logger().info(f'Initial state: {self.current_state.value}')
    
    def odom_callback(self, msg):
        """Update current pose từ odometry"""
        self.current_pose = msg.pose.pose
        self.current_altitude = msg.pose.pose.position.z
        self.last_odom_time = time.time()
        
        # Check vision status
        self.vision_ok = True  # Assume OK if receiving odometry
    
    def command_callback(self, msg):
        """Handle external commands"""
        command = msg.data.lower()
        
        if command == "takeoff":
            if self.current_state == FlightState.IDLE:
                self.transition_to(FlightState.TAKEOFF)
        elif command == "land":
            if self.current_state in [FlightState.NAVIGATION, FlightState.HOVER]:
                self.transition_to(FlightState.LANDING)
        elif command == "emergency":
            self.transition_to(FlightState.EMERGENCY)
        elif command == "rtl":
            if self.current_state in [FlightState.NAVIGATION, FlightState.HOVER]:
                self.transition_to(FlightState.RETURN_TO_LAUNCH)
        elif command == "idle":
            if self.current_state in [FlightState.HOVER, FlightState.LANDING]:
                self.transition_to(FlightState.IDLE)
    
    def transition_to(self, new_state: FlightState):
        """Transition to new state"""
        if new_state == self.current_state:
            return
        
        self.get_logger().info(f'State transition: {self.current_state.value} -> {new_state.value}')
        self.current_state = new_state
        self.state_start_time = time.time()
        
        # Publish state
        msg = String()
        msg.data = self.current_state.value
        self.state_pub.publish(msg)
    
    def state_machine_logic(self):
        """Main state machine logic"""
        elapsed_time = time.time() - self.state_start_time
        
        # Safety checks
        safety_ok = self.check_safety()
        
        if not safety_ok and self.current_state != FlightState.EMERGENCY:
            self.transition_to(FlightState.EMERGENCY)
            return
        
        # State-specific logic
        if self.current_state == FlightState.IDLE:
            self.handle_idle()
        
        elif self.current_state == FlightState.TAKEOFF:
            self.handle_takeoff(elapsed_time)
        
        elif self.current_state == FlightState.NAVIGATION:
            self.handle_navigation()
        
        elif self.current_state == FlightState.HOVER:
            self.handle_hover()
        
        elif self.current_state == FlightState.LANDING:
            self.handle_landing(elapsed_time)
        
        elif self.current_state == FlightState.RETURN_TO_LAUNCH:
            self.handle_rtl()
        
        elif self.current_state == FlightState.EMERGENCY:
            self.handle_emergency()
    
    def check_safety(self) -> bool:
        """Check safety conditions"""
        # Check altitude
        if self.current_altitude > self.max_altitude:
            self.get_logger().warn(f'Altitude too high: {self.current_altitude}m')
            return False
        
        # Check vision
        stale_odom = False
        if self.last_odom_time is None:
            stale_odom = True
        else:
            stale_odom = (time.time() - self.last_odom_time) > self.odom_timeout

        if stale_odom:
            self.vision_ok = False
        if not self.vision_ok:
            self.get_logger().warn('Vision system not OK')
            # Don't fail immediately, but warn
        
        # Check PX4 connection
        if not self.px4_connected:
            self.get_logger().warn('PX4 not connected')
            # Don't fail immediately
        
        # Publish safety status
        msg = Bool()
        msg.data = not stale_odom and self.current_altitude <= self.max_altitude
        self.safety_pub.publish(msg)
        
        return True
    
    def handle_idle(self):
        """Handle IDLE state"""
        # Wait for takeoff command
        pass
    
    def handle_takeoff(self, elapsed_time: float):
        """Handle TAKEOFF state"""
        if self.current_pose is None:
            return
        
        # Check if reached target altitude
        if self.current_altitude >= self.target_altitude * 0.9:  # 90% of target
            self.get_logger().info('Takeoff complete')
            self.transition_to(FlightState.HOVER)
            return
        
        # Check timeout
        if elapsed_time > self.takeoff_timeout:
            self.get_logger().warn('Takeoff timeout')
            self.transition_to(FlightState.EMERGENCY)
            return
        
        # Send takeoff command to PX4 (via topic)
        # In real implementation, send MAVLink command
    
    def handle_navigation(self):
        """Handle NAVIGATION state"""
        if not self.waypoints:
            self.transition_to(FlightState.HOVER)
            return
        
        # Check if reached current waypoint
        if self.current_waypoint_index < len(self.waypoints):
            waypoint = self.waypoints[self.current_waypoint_index]
            distance = self.distance_to_waypoint(waypoint)
            
            if distance < self.waypoint_tolerance:
                self.get_logger().info(f'Reached waypoint {self.current_waypoint_index}')
                self.current_waypoint_index += 1
                
                if self.current_waypoint_index >= len(self.waypoints):
                    self.get_logger().info('All waypoints reached')
                    self.transition_to(FlightState.HOVER)
            else:
                # Navigate to waypoint
                self.navigate_to_waypoint(waypoint)
        else:
            self.transition_to(FlightState.HOVER)
    
    def handle_hover(self):
        """Handle HOVER state"""
        # Maintain current position
        # Wait for navigation command or landing
        pass
    
    def handle_landing(self, elapsed_time: float):
        """Handle LANDING state"""
        if self.current_pose is None:
            return
        
        # Check if landed
        if self.current_altitude <= self.emergency_landing_altitude:
            self.get_logger().info('Landed')
            self.transition_to(FlightState.IDLE)
            return
        
        # Check timeout
        if elapsed_time > self.landing_timeout:
            self.get_logger().warn('Landing timeout')
            self.transition_to(FlightState.EMERGENCY)
            return
        
        # Send landing command to PX4
    
    def handle_rtl(self):
        """Handle RETURN_TO_LAUNCH state"""
        # Navigate to launch position (0, 0, target_altitude)
        launch_waypoint = [0.0, 0.0, self.target_altitude]
        distance = self.distance_to_waypoint(launch_waypoint)
        
        if distance < self.waypoint_tolerance:
            self.get_logger().info('Returned to launch')
            self.transition_to(FlightState.HOVER)
        else:
            self.navigate_to_waypoint(launch_waypoint)
    
    def handle_emergency(self):
        """Handle EMERGENCY state"""
        # Emergency landing
        if self.current_altitude > self.emergency_landing_altitude:
            # Land immediately
            pass
        else:
            # Already landed, go to IDLE
            self.transition_to(FlightState.IDLE)
    
    def distance_to_waypoint(self, waypoint) -> float:
        """Calculate distance to waypoint"""
        if self.current_pose is None:
            return float('inf')
        
        dx = waypoint[0] - self.current_pose.position.x
        dy = waypoint[1] - self.current_pose.position.y
        dz = waypoint[2] - self.current_pose.position.z
        
        return (dx*dx + dy*dy + dz*dz) ** 0.5
    
    def navigate_to_waypoint(self, waypoint):
        """Navigate to waypoint (publish command)"""
        # In real implementation, publish to PX4 control topic
        # For now, just log
        self.get_logger().debug(f'Navigating to waypoint: {waypoint}')
    
    def set_waypoints(self, waypoints: list):
        """Set mission waypoints"""
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        
        if waypoints and self.current_state == FlightState.HOVER:
            self.transition_to(FlightState.NAVIGATION)


def main(args=None):
    rclpy.init(args=args)
    node = StateMachine()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

