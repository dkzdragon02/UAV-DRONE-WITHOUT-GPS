import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
import time
import threading

try:
    from pymavlink import mavutil
    PYMAVLINK_AVAILABLE = True
except ImportError:
    PYMAVLINK_AVAILABLE = False
    print("Warning: pymavlink not available. Install with: pip3 install pymavlink")

class PX4MAVLinkBridge(Node):  
    def __init__(self):
        super().__init__('px4_mavlink_bridge')
        
        if not PYMAVLINK_AVAILABLE:
            self.get_logger().error('pymavlink not available! Cannot connect to PX4.')
            return
        
        self.declare_parameter('vision_odom_topic', '/uav/vision/odometry')
        self.declare_parameter('px4_connection', 'udp:127.0.0.1:18570')
        self.declare_parameter('send_rate', 30.0)   # Hz
        
        vision_topic = self.get_parameter('vision_odom_topic').value
        connection_str = self.get_parameter('px4_connection').value
        send_rate = self.get_parameter('send_rate').value
        
        self.mav_connection = None
        self.connected = False
        self.lock = threading.Lock()
        self.last_odom = None
        self.cmd_vel = [0.0, 0.0, 0.0]              # vx, vy, vz (LOCAL_NED)
        
        self.odom_sub = self.create_subscription(
            Odometry,
            vision_topic,
            self.odom_callback,
            10
        )
        
        self.status_pub = self.create_publisher(String, '/px4_mavlink/status', 10)
        self.connected_pub = self.create_publisher(Bool, '/px4_mavlink/connected', 10)
        
        self.set_mode_sub = self.create_subscription(
            String,
            '/px4_mavlink/set_mode',
            self.set_mode_callback,
            10
        )
        
        self.arm_sub = self.create_subscription(
            Bool,
            '/px4_mavlink/arm',
            self.arm_callback,
            10
        )

        self.vel_sp_sub = self.create_subscription(
            Twist,
            '/px4_mavlink/setpoint_vel',
            self.velocity_setpoint_callback,
            10
        )
        
        self.send_timer = self.create_timer(1.0 / send_rate, self.send_vision_position)
        self.status_timer = self.create_timer(1.0, self.publish_connection_status)
        self.vel_timer = self.create_timer(0.05, self.send_velocity_setpoint)
        self.connect_to_px4(connection_str)
        self.get_logger().info('PX4 MAVLink Bridge started')
        self.get_logger().info(f'Subscribing to: {vision_topic}')
        self.get_logger().info(f'Connection: {connection_str}')
    
    def connect_to_px4(self, connection_str):
        max_retries = 10
        retry_delay = 2  # seconds
        
        if connection_str.startswith('udp:'):
            px4_ports = ['18570', '14580', '14280', '13030']
            for port in px4_ports:
                if f':{port}' in connection_str:
                    alt_connection = connection_str.replace('udp:', 'udpout:')
                    self.get_logger().info(f'Port {port} may be in use, using udpout: (send-only mode)')
                    connection_str = alt_connection
                    break
        
        for attempt in range(max_retries):
            try:
                self.get_logger().info(
                    f'Connecting to PX4: {connection_str} (attempt {attempt + 1}/{max_retries})'
                )
                self.mav_connection = mavutil.mavlink_connection(connection_str)
                self.mav_connection.wait_heartbeat(timeout=10)
                self.connected = True
                self.get_logger().info('Connected to PX4!')
                self.publish_status('Connected to PX4')
                
                heartbeat_thread = threading.Thread(target=self.send_heartbeat, daemon=True)
                heartbeat_thread.start()
                return
                
            except Exception as e:
                self.get_logger().warn(f'Connection attempt {attempt + 1} failed: {e}')
                if attempt < max_retries - 1:
                    self.get_logger().info(f'Retrying in {retry_delay} seconds...')
                    time.sleep(retry_delay)
                else:
                    self.get_logger().error(
                        f'Failed to connect to PX4 after {max_retries} attempts: {e}'
                    )
                    self.connected = False
                    self.publish_status(f'Connection failed: {e}')
    
    def send_heartbeat(self):
        while rclpy.ok() and self.connected:
            try:
                if self.mav_connection:
                    self.mav_connection.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_GCS,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                        0, 0, 0
                    )
                time.sleep(1)
            except Exception as e:
                self.get_logger().error(f'Error sending heartbeat: {e}')
                self.connected = False
                break
    
    def odom_callback(self, msg):
        with self.lock:
            self.last_odom = msg

    def velocity_setpoint_callback(self, msg: Twist):
        with self.lock:
            self.cmd_vel[0] = msg.linear.x
            self.cmd_vel[1] = msg.linear.y
            self.cmd_vel[2] = msg.linear.z
    
    def send_vision_position(self):
        if not self.connected or not self.mav_connection:
            return
        
        with self.lock:
            if self.last_odom is None:
                return
            
            odom = self.last_odom
            current_time = time.time()
            usec = int(current_time * 1e6)
            x = odom.pose.pose.position.x
            y = odom.pose.pose.position.y
            z = odom.pose.pose.position.z
            q = odom.pose.pose.orientation
            
            import math
            sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
            cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
            roll = math.atan2(sinr_cosp, cosr_cosp)
            
            sinp = 2.0 * (q.w * q.y - q.z * q.x)
            if abs(sinp) >= 1.0:
                pitch = math.copysign(math.pi / 2.0, sinp)
            else:
                pitch = math.asin(sinp)
            
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            vx = odom.twist.twist.linear.x if odom.twist else 0.0
            vy = odom.twist.twist.linear.y if odom.twist else 0.0
            vz = odom.twist.twist.linear.z if odom.twist else 0.0
            
            try:
                covariance = [0.0] * 21
                if hasattr(odom.pose, 'covariance') and len(odom.pose.covariance) >= 36:
                    cov_6x6 = odom.pose.covariance
                    idx = 0
                    for row in range(6):
                        for col in range(row, 6):
                            covariance[idx] = float(cov_6x6[row * 6 + col])
                            idx += 1
                else:
                    diag_indices = [0, 6, 11, 15, 18, 20]
                    for i in diag_indices:
                        covariance[i] = 0.01
                
                self.mav_connection.mav.vision_position_estimate_send(
                    usec,           # us: Timestamp (microseconds, synced to UNIX time or since system boot)
                    x,              # x: Global X position (FIXED_FRAME)
                    y,              # y: Global Y position
                    z,              # z: Global Z position
                    roll,           # roll: Roll angle (rad)
                    pitch,          # pitch: Pitch angle (rad)
                    yaw,            # yaw: Yaw angle (rad)
                    covariance      # covariance: Upper-triangle of 6x6 pose cross-covariance (21 elements) (FIXED_FRAME)
                )
                
                if abs(vx) > 0.001 or abs(vy) > 0.001 or abs(vz) > 0.001:
                    self.mav_connection.mav.vision_speed_estimate_send(
                        usec,       # us: Timestamp (microseconds, synced to UNIX time or since system boot)
                        vx,         # x: Global X speed
                        vy,         # y: Global Y speed
                        vz,         # z: Global Z speed
                        [0.0] * 9   # covariance: Row-major representation of 3x3 cross-covariance matrix
                    )
                
            except Exception as e:
                self.get_logger().error(f'Error sending vision position: {e}')

    def send_velocity_setpoint(self):
        if not self.connected or not self.mav_connection:
            return

        with self.lock:
            vx, vy, vz = self.cmd_vel

        if abs(vx) < 1e-3 and abs(vy) < 1e-3 and abs(vz) < 1e-3:
            return

        try:

            time_boot_ms = int(time.time() * 1000) & 0xFFFFFFFF             # Thời gian ước lượng (ms)
            frame = mavutil.mavlink.MAV_FRAME_LOCAL_NED                     # LOCAL_NED frame, dùng velocity only
            type_mask = 0b0000111111000111

            self.mav_connection.mav.set_position_target_local_ned_send(
                time_boot_ms,
                self.mav_connection.target_system,
                self.mav_connection.target_component,
                frame,
                type_mask,
                0.0, 0.0, 0.0,   # x, y, z (ignored)
                float(vx),       # vx
                float(vy),       # vy
                float(vz),       # vz
                0.0, 0.0, 0.0,   # ax, ay, az (ignored)
                0.0,             # yaw (ignored)
                0.0              # yaw_rate (ignored)
            )
        except Exception as e:
            self.get_logger().error(f'Error sending velocity setpoint: {e}')
    
    def set_mode_callback(self, msg):
        if not self.connected or not self.mav_connection:
            self.get_logger().warn('Not connected to PX4')
            return
        
        mode = msg.data.strip()
        self.get_logger().info(f'Setting flight mode to: {mode}')
        
        try:
            base_mode = mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
            custom_mode = 0
            
            if mode.upper() == 'OFFBOARD':
                custom_mode = 6  # OFFBOARD mode
            elif mode.upper() == 'POSCTL':
                custom_mode = 8  # POSCTL mode
            elif mode.upper() == 'ALTCTL':
                custom_mode = 2  # ALTCTL mode
            
            self.mav_connection.mav.set_mode_send(
                self.mav_connection.target_system,
                base_mode,
                custom_mode
            )
            
            self.get_logger().info(f'Flight mode set to: {mode}')
            
        except Exception as e:
            self.get_logger().error(f'Error setting mode: {e}')
    
    def arm_callback(self, msg):
        if not self.connected or not self.mav_connection:
            self.get_logger().warn('Not connected to PX4')
            return
        
        arm = msg.data
        action = 'ARM' if arm else 'DISARM'
        self.get_logger().info(f'{action} vehicle')
        
        try:
            param1 = 1.0 if arm else 0.0    # 1 = arm, 0 = disarm
            param2 = 0.0                    # Force disarm (not used for arm)
            
            self.mav_connection.mav.command_long_send(
                self.mav_connection.target_system,
                self.mav_connection.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 
                param1,
                param2,
                0.0, 0.0, 0.0, 0.0, 0.0
            )
            
            self.get_logger().info(f'Vehicle {action.lower()} command sent')
            
        except Exception as e:
            self.get_logger().error(f'Error {action.lower()}ing: {e}')
    
    def publish_status(self, status_msg):
        msg = String()
        msg.data = status_msg
        self.status_pub.publish(msg)
        self.publish_connection_status()
    
    def publish_connection_status(self):
        connected_msg = Bool()
        connected_msg.data = self.connected
        self.connected_pub.publish(connected_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PX4MAVLinkBridge()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.mav_connection:
            node.mav_connection.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

