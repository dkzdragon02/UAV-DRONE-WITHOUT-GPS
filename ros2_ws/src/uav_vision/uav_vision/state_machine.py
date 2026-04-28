import rclpy
from rclpy.node import Node
from enum import Enum
from std_msgs.msg import String, Bool, Float64
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State as MavrosState
import time
from uav_vision.qos_profiles import mavros_qos as make_mavros_qos
from uav_vision.geofencing import Geofence, ViolationType

class FlightState(Enum):
    IDLE = "idle"
    TAKEOFF = "takeoff"
    NAVIGATION = "navigation"
    LANDING = "landing"
    EMERGENCY = "emergency"
    RETURN_TO_LAUNCH = "rtl"
    HOVER = "hover"
    MANUAL_CONTROL = "manual_control"

class StateMachine(Node): 
    def __init__(self):
        super().__init__('state_machine')
        
        self.current_state = FlightState.IDLE
        self.state_start_time = time.time()
        self.declare_parameter('odom_topic', '/uav/vision/odometry')
        self.declare_parameter('fused_odom_topic', '/uav/fused_odometry')
        self.declare_parameter('mavros_pose_topic', '/mavros/local_position/pose')
        self.declare_parameter('target_altitude', 2.0)          # meters
        self.declare_parameter('takeoff_timeout', 30.0)         # seconds
        self.declare_parameter('landing_timeout', 30.0)         # seconds
        self.declare_parameter('emergency_landing_altitude', 0.5)  # meters
        self.declare_parameter('odom_timeout', 2.0)             # seconds
        self.declare_parameter('auto_takeoff', False)           # auto takeoff on ARM+OFFBOARD
        
        odom_topic = self.get_parameter('odom_topic').value
        fused_odom_topic = self.get_parameter('fused_odom_topic').value
        mavros_pose_topic = self.get_parameter('mavros_pose_topic').value
        self.target_altitude = self.get_parameter('target_altitude').value
        self.takeoff_timeout = self.get_parameter('takeoff_timeout').value
        self.landing_timeout = self.get_parameter('landing_timeout').value
        self.emergency_landing_altitude = self.get_parameter('emergency_landing_altitude').value
        self.odom_timeout = self.get_parameter('odom_timeout').value
        self.auto_takeoff = self.get_parameter('auto_takeoff').value
        self.current_pose = None
        self.current_altitude = 0.0
        self.fused_altitude = 0.0
        self.has_fused_odom = False
        self.last_odom_time = None
        self.waypoints = []
        self.current_waypoint_index = 0
        self.waypoint_tolerance = 0.5       # meters
        self.max_altitude = 10.0            # meters
        self.min_battery_voltage = 10.5     # volts (example)
        self.vision_ok = False
        self.px4_connected = False
        self.px4_armed = False
        self.px4_mode = ""
        self.auto_takeoff_triggered = False
        self.obstacle_emergency_stop = False
        self.nearest_obstacle_dist = float('inf')
        self.obstacle_avoidance_enabled = True
        self.declare_parameter('geofence_radius', 50.0)
        self.declare_parameter('geofence_max_altitude', 30.0)
        self.declare_parameter('geofence_warning_margin', 2.0)
        self.declare_parameter('geofence_config_path', '')
        
        geofence_radius = self.get_parameter('geofence_radius').value
        geofence_max_alt = self.get_parameter('geofence_max_altitude').value
        geofence_warning = self.get_parameter('geofence_warning_margin').value
        geofence_config = self.get_parameter('geofence_config_path').value
        
        self.geofence = Geofence(
            max_radius=geofence_radius,
            max_altitude=geofence_max_alt,
            warning_margin=geofence_warning
        )
        
        if geofence_config:
            self.geofence.load_from_config(geofence_config)
            self.get_logger().info(f'Geofence config loaded from: {geofence_config}')
        
        self.get_logger().info(
            f'Geofence active: radius={geofence_radius}m, max_alt={geofence_max_alt}m'
        )
        
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 10
        )
        
        self.fused_odom_sub = self.create_subscription(
            Odometry, fused_odom_topic, self.fused_odom_callback, 10
        )
        
        mavros_qos = make_mavros_qos()
        
        self.mavros_pose_sub = self.create_subscription(
            PoseStamped, mavros_pose_topic, self.mavros_pose_callback, mavros_qos
        )
        
        self.mavros_state_sub = self.create_subscription(
            MavrosState, '/mavros/state', self.mavros_state_callback, mavros_qos
        )
        
        self.command_sub = self.create_subscription(
            String, '/uav/state_machine/command', self.command_callback, 10
        )
        
        self.obs_estop_sub = self.create_subscription(
            Bool, '/uav/obstacle_avoidance/emergency_stop',
            self.obstacle_estop_callback, 10
        )
        self.obs_dist_sub = self.create_subscription(
            Float64, '/uav/obstacle_avoidance/nearest_obstacle_dist',
            self.obstacle_dist_callback, 10
        )
         
        self.state_pub = self.create_publisher(String, '/uav/state_machine/state', 10)
        self.safety_pub = self.create_publisher(Bool, '/uav/state_machine/safety_ok', 10)
        self.status_pub = self.create_publisher(String, '/uav/state_machine/status', 10)
        self.state_timer = self.create_timer(0.1, self.state_machine_logic)   # 10 Hz
        self.status_timer = self.create_timer(2.0, self.publish_status)       # 0.5 Hz
        self.get_logger().info('State Machine started')
        self.get_logger().info(f'Initial state: {self.current_state.value}')
        self.get_logger().info(f'Auto-takeoff: {self.auto_takeoff}')
    
    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        if not self.px4_connected and not self.has_fused_odom:
            self.current_altitude = msg.pose.pose.position.z
        self.last_odom_time = time.time()
        self.vision_ok = True
    
    def fused_odom_callback(self, msg):
        self.has_fused_odom = True
        self.fused_altitude = msg.pose.pose.position.z
        if not self.px4_connected:
            self.current_altitude = self.fused_altitude
        self.current_pose = msg.pose.pose
        self.last_odom_time = time.time()
        self.vision_ok = True
    
    def mavros_pose_callback(self, msg):
        self.current_pose = msg.pose
        self.current_altitude = msg.pose.position.z
        self.last_odom_time = time.time()
    
    def mavros_state_callback(self, msg):
        was_armed = self.px4_armed
        self.px4_connected = msg.connected
        self.px4_armed = msg.armed
        self.px4_mode = msg.mode
        
        if (self.auto_takeoff and not self.auto_takeoff_triggered
                and self.px4_armed and self.px4_mode == "OFFBOARD"
                and self.current_state == FlightState.IDLE):
            self.get_logger().info('Auto-takeoff triggered: PX4 armed + OFFBOARD detected')
            self.transition_to(FlightState.TAKEOFF)
            self.auto_takeoff_triggered = True
    
    def obstacle_estop_callback(self, msg):
        was_stopped = self.obstacle_emergency_stop
        self.obstacle_emergency_stop = msg.data
        
        if msg.data and not was_stopped:
            self.get_logger().warn(
                f'Obstacle E-STOP! Nearest obstacle: {self.nearest_obstacle_dist:.2f}m'
            )
            if self.current_state in (
                FlightState.NAVIGATION, FlightState.MANUAL_CONTROL
            ):
                self.get_logger().warn('Transitioning to HOVER due to obstacle E-STOP')
                self.transition_to(FlightState.HOVER)
    
    def obstacle_dist_callback(self, msg):
        self.nearest_obstacle_dist = msg.data
    
    def command_callback(self, msg):
        command = msg.data.lower().strip()
        
        self.get_logger().info(f'Received command: "{command}"')
        
        if command == "takeoff":
            if self.current_state == FlightState.IDLE:
                self.transition_to(FlightState.TAKEOFF)
            else:
                self.get_logger().warn(f'Cannot takeoff from state: {self.current_state.value}')
                
        elif command == "land":
            if self.current_state in [FlightState.NAVIGATION, FlightState.HOVER, FlightState.TAKEOFF, FlightState.MANUAL_CONTROL]:
                self.transition_to(FlightState.LANDING)
            else:
                self.get_logger().warn(f'Cannot land from state: {self.current_state.value}')
                
        elif command == "emergency":
            self.transition_to(FlightState.EMERGENCY)
            
        elif command == "rtl":
            if self.current_state in [FlightState.NAVIGATION, FlightState.HOVER, FlightState.MANUAL_CONTROL]:
                self.transition_to(FlightState.RETURN_TO_LAUNCH)
                
        elif command == "hover":
            if self.current_state in [FlightState.NAVIGATION, FlightState.TAKEOFF, FlightState.MANUAL_CONTROL]:
                self.transition_to(FlightState.HOVER)
                
        elif command == "navigate":
            if self.current_state == FlightState.HOVER:
                self.transition_to(FlightState.NAVIGATION)

        elif command == "manual":
            if self.current_state in [FlightState.HOVER, FlightState.NAVIGATION, FlightState.TAKEOFF]:
                self.transition_to(FlightState.MANUAL_CONTROL)
            else:
                self.get_logger().warn(f'Cannot enter manual from state: {self.current_state.value}')
                
        elif command == "idle":
            if self.current_state in [FlightState.HOVER, FlightState.LANDING]:
                self.transition_to(FlightState.IDLE)
                
        else:
            self.get_logger().warn(f'Unknown command: {command}')
    
    def transition_to(self, new_state: FlightState):
        if new_state == self.current_state:
            return
        
        self.get_logger().info(
            f'═══ State transition: {self.current_state.value} → {new_state.value} ═══'
        )
        self.current_state = new_state
        self.state_start_time = time.time()
        
        msg = String()
        msg.data = self.current_state.value
        self.state_pub.publish(msg)
     
    def state_machine_logic(self):
        elapsed_time = time.time() - self.state_start_time
        state_msg = String()
        state_msg.data = self.current_state.value
        self.state_pub.publish(state_msg)
        safety_ok = self.check_safety()
        
        if not safety_ok and self.current_state not in [FlightState.EMERGENCY, FlightState.IDLE]:
            self.transition_to(FlightState.EMERGENCY)
            
            return
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
        elif self.current_state == FlightState.MANUAL_CONTROL:
            self.handle_manual_control()
        elif self.current_state == FlightState.EMERGENCY:
            self.handle_emergency()
    
    def check_safety(self) -> bool:
        if self.current_altitude > self.max_altitude:
            self.get_logger().warn(f'Altitude too high: {self.current_altitude:.1f}m')
            return False

        in_ground_state = (
            self.current_state == FlightState.IDLE
            or (self.current_state == FlightState.TAKEOFF and self.current_altitude < 0.5)
        )
        
        if self.current_pose is not None and not in_ground_state:
            gf_result = self.geofence.check_position(
                self.current_pose.position.x,
                self.current_pose.position.y,
                self.current_altitude
            )
            
            if not gf_result.is_safe:
                self.get_logger().error(
                    f'GEOFENCE VIOLATION: {gf_result.violation_type.value} — {gf_result.message}'
                )
                return False
            
            if gf_result.distance_to_boundary < self.geofence.warning_margin:
                self.get_logger().warn(
                    f'Near geofence boundary: {gf_result.distance_to_boundary:.1f}m remaining',
                    throttle_duration_sec=5.0
                )
                if self.current_state == FlightState.NAVIGATION:
                    self.get_logger().warn('Near boundary during navigation — switching to RTL')
                    self.transition_to(FlightState.RETURN_TO_LAUNCH)
        
        stale_odom = False
        if self.last_odom_time is None:
            stale_odom = True
        else:
            stale_odom = (time.time() - self.last_odom_time) > self.odom_timeout

        if stale_odom:
            self.vision_ok = False
        if not self.vision_ok:
            self.get_logger().warn('Vision system not OK', throttle_duration_sec=5.0)
        
        msg = Bool()
        msg.data = not stale_odom and self.current_altitude <= self.max_altitude
        self.safety_pub.publish(msg)
        
        return True
    
    def handle_idle(self):
        pass
    
    def handle_takeoff(self, elapsed_time: float):
        if self.current_pose is None:
            return

        if self.current_altitude >= self.target_altitude * 0.9:
            self.get_logger().info(
                f'Takeoff complete at {self.current_altitude:.2f}m'
            )
            self.transition_to(FlightState.HOVER)
            return
        
        if elapsed_time > self.takeoff_timeout:
            self.get_logger().warn('⚠ Takeoff timeout!')
            self.transition_to(FlightState.EMERGENCY)
            return
        if int(elapsed_time) % 5 == 0 and elapsed_time > 0:
            self.get_logger().info(
                f'Takeoff progress: {self.current_altitude:.2f}m / {self.target_altitude}m '
                f'({elapsed_time:.0f}s)',
                throttle_duration_sec=4.5
            )
    
    def handle_navigation(self):
        if not self.waypoints:
            pass
    
    def handle_hover(self):
        pass

    def handle_manual_control(self):
        pass
    
    def handle_landing(self, elapsed_time: float):
        if self.current_pose is None:
            return
        
        if self.current_altitude <= self.emergency_landing_altitude:
            self.get_logger().info('Landed')
            self.transition_to(FlightState.IDLE)
            return
        
        if elapsed_time > self.landing_timeout:
            self.get_logger().warn('Landing timeout!')
            self.transition_to(FlightState.EMERGENCY)
            return
    
    def handle_rtl(self):
        if self.current_pose is not None:
            dx = self.current_pose.position.x
            dy = self.current_pose.position.y
            distance = (dx*dx + dy*dy) ** 0.5
            
            if distance < self.waypoint_tolerance:
                self.get_logger().info('Returned to launch')
                self.transition_to(FlightState.HOVER)
    
    def handle_emergency(self):
        if self.current_altitude <= self.emergency_landing_altitude:
            self.get_logger().info('Emergency landing complete')
            self.transition_to(FlightState.IDLE)
    
    def publish_status(self):
        elapsed = time.time() - self.state_start_time
        obs_dist_str = f'{self.nearest_obstacle_dist:.1f}m' if self.nearest_obstacle_dist < 100 else 'clear'
        msg = String()
        msg.data = (
            f"state={self.current_state.value} | "
            f"alt={self.current_altitude:.2f}m | "
            f"px4={'CONN' if self.px4_connected else 'DISC'} | "
            f"armed={'YES' if self.px4_armed else 'NO'} | "
            f"mode={self.px4_mode} | "
            f"vision={'OK' if self.vision_ok else 'FAIL'} | "
            f"obs={'ESTOP' if self.obstacle_emergency_stop else obs_dist_str} | "
            f"elapsed={elapsed:.0f}s"
        )
        self.status_pub.publish(msg)
    
    def set_waypoints(self, waypoints: list):
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
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
