#!/usr/bin/env python3
"""
ROS2 Node kết nối trực tiếp với PX4 qua MAVLink (không cần ROS1 MAVROS)
Gửi vision position estimate và các lệnh điều khiển trực tiếp tới PX4
"""

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
    """Bridge trực tiếp giữa ROS2 Vision và PX4 qua MAVLink"""
    
    def __init__(self):
        super().__init__('px4_mavlink_bridge')
        
        if not PYMAVLINK_AVAILABLE:
            self.get_logger().error('pymavlink not available! Cannot connect to PX4.')
            return
        
        # Parameters
        self.declare_parameter('vision_odom_topic', '/uav/vision/odometry')
        self.declare_parameter('px4_connection', 'udp:127.0.0.1:18570')
        self.declare_parameter('send_rate', 30.0)  # Hz
        
        vision_topic = self.get_parameter('vision_odom_topic').value
        connection_str = self.get_parameter('px4_connection').value
        send_rate = self.get_parameter('send_rate').value
        
        # MAVLink connection
        self.mav_connection = None
        self.connected = False
        
        # Threading
        self.lock = threading.Lock()
        self.last_odom = None
        self.cmd_vel = [0.0, 0.0, 0.0]  # vx, vy, vz (LOCAL_NED)
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            vision_topic,
            self.odom_callback,
            10
        )
        
        # Publishers for status
        self.status_pub = self.create_publisher(String, '/px4_mavlink/status', 10)
        self.connected_pub = self.create_publisher(Bool, '/px4_mavlink/connected', 10)
        
        # Services/Commands (via topics for simplicity)
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

        # Velocity setpoint from higher-level planner / obstacle avoidance
        self.vel_sp_sub = self.create_subscription(
            Twist,
            '/px4_mavlink/setpoint_vel',
            self.velocity_setpoint_callback,
            10
        )
        
        # Timer for sending vision position estimate
        self.send_timer = self.create_timer(1.0 / send_rate, self.send_vision_position)
        
        # Timer for publishing connection status (1 Hz)
        self.status_timer = self.create_timer(1.0, self.publish_connection_status)

        # Timer for sending velocity setpoints (OFFBOARD control), 20 Hz
        self.vel_timer = self.create_timer(0.05, self.send_velocity_setpoint)
        
        # Connect to PX4
        self.connect_to_px4(connection_str)
        
        self.get_logger().info('PX4 MAVLink Bridge started')
        self.get_logger().info(f'Subscribing to: {vision_topic}')
        self.get_logger().info(f'Connection: {connection_str}')
    
    def connect_to_px4(self, connection_str):
        """Kết nối với PX4 qua MAVLink"""
        max_retries = 10
        retry_delay = 2  # seconds
        
        # Với UDP, pymavlink sẽ bind vào port trong connection string
        # Nếu port đã bị PX4 bind (18570, 14580, 14280, 13030), dùng udpout: để chỉ gửi
        if connection_str.startswith('udp:'):
            px4_ports = ['18570', '14580', '14280', '13030']
            for port in px4_ports:
                if f':{port}' in connection_str:
                    # Dùng udpout: để chỉ gửi (không bind vào port, tránh conflict)
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
                
                # Start heartbeat thread
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
        """Gửi heartbeat để duy trì kết nối"""
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
        """Callback khi nhận odometry từ vision"""
        with self.lock:
            self.last_odom = msg

    def velocity_setpoint_callback(self, msg: Twist):
        """Nhận setpoint vận tốc từ obstacle_avoidance_node / planner"""
        with self.lock:
            self.cmd_vel[0] = msg.linear.x
            self.cmd_vel[1] = msg.linear.y
            self.cmd_vel[2] = msg.linear.z
    
    def send_vision_position(self):
        """Gửi VISION_POSITION_ESTIMATE message tới PX4"""
        if not self.connected or not self.mav_connection:
            return
        
        with self.lock:
            if self.last_odom is None:
                return
            
            odom = self.last_odom
            current_time = time.time()
            
            # Convert ROS time to milliseconds since boot (approximate)
            # PX4 expects usec timestamp, we'll use current time
            usec = int(current_time * 1e6)
            
            # Extract position (assume NED frame)
            x = odom.pose.pose.position.x
            y = odom.pose.pose.position.y
            z = odom.pose.pose.position.z
            
            # Extract orientation quaternion
            q = odom.pose.pose.orientation
            qx = q.x
            qy = q.y
            qz = q.z
            qw = q.w
            
            # Extract velocity if available
            vx = odom.twist.twist.linear.x if odom.twist else 0.0
            vy = odom.twist.twist.linear.y if odom.twist else 0.0
            vz = odom.twist.twist.linear.z if odom.twist else 0.0
            
            try:
                # Send VISION_POSITION_ESTIMATE message
                # MAVLink message ID: 102 (VISION_POSITION_ESTIMATE)
                self.mav_connection.mav.vision_position_estimate_send(
                    usec,  # us: Timestamp (microseconds, synced to UNIX time or since system boot)
                    x,     # x: Global X position
                    y,     # y: Global Y position
                    z,     # z: Global Z position
                    qx,    # roll: Roll angle
                    qy,    # pitch: Pitch angle
                    qz,    # yaw: Yaw angle
                    [0.0] * 4  # covariance: Row-major representation of pose 6x6 cross-covariance matrix
                )
                
                # Also send VISION_SPEED_ESTIMATE if velocity is available
                if abs(vx) > 0.001 or abs(vy) > 0.001 or abs(vz) > 0.001:
                    self.mav_connection.mav.vision_speed_estimate_send(
                        usec,  # us: Timestamp (microseconds)
                        vx,    # x: Global X speed
                        vy,    # y: Global Y speed
                        vz,    # z: Global Z speed
                        [0.0] * 9  # covariance: Row-major representation of 3x3 cross-covariance matrix
                    )
                
            except Exception as e:
                self.get_logger().error(f'Error sending vision position: {e}')

    def send_velocity_setpoint(self):
        """Gửi SET_POSITION_TARGET_LOCAL_NED với vận tốc từ /px4_mavlink/setpoint_vel"""
        if not self.connected or not self.mav_connection:
            return

        with self.lock:
            vx, vy, vz = self.cmd_vel

        # Nếu vận tốc rất nhỏ thì không cần gửi (giảm bớt traffic)
        if abs(vx) < 1e-3 and abs(vy) < 1e-3 and abs(vz) < 1e-3:
            return

        try:
            # Thời gian ước lượng (ms)
            time_boot_ms = int(time.time() * 1000) & 0xFFFFFFFF

            # LOCAL_NED frame, dùng velocity only
            frame = mavutil.mavlink.MAV_FRAME_LOCAL_NED

            type_mask = 0b0000111111000111
            # Ignore position, acceleration, yaw, yaw_rate. Chỉ dùng vx, vy, vz.

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
        """Set flight mode"""
        if not self.connected or not self.mav_connection:
            self.get_logger().warn('Not connected to PX4')
            return
        
        mode = msg.data.strip()
        self.get_logger().info(f'Setting flight mode to: {mode}')
        
        try:
            # For custom modes, use COMMAND_LONG
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
        """Arm/Disarm vehicle"""
        if not self.connected or not self.mav_connection:
            self.get_logger().warn('Not connected to PX4')
            return
        
        arm = msg.data
        action = 'ARM' if arm else 'DISARM'
        self.get_logger().info(f'{action} vehicle')
        
        try:
            # MAV_CMD_COMPONENT_ARM_DISARM
            param1 = 1.0 if arm else 0.0  # 1 = arm, 0 = disarm
            param2 = 0.0  # Force disarm (not used for arm)
            
            self.mav_connection.mav.command_long_send(
                self.mav_connection.target_system,
                self.mav_connection.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,  # confirmation
                param1,
                param2,
                0.0, 0.0, 0.0, 0.0, 0.0
            )
            
            self.get_logger().info(f'Vehicle {action.lower()} command sent')
            
        except Exception as e:
            self.get_logger().error(f'Error {action.lower()}ing: {e}')
    
    def publish_status(self, status_msg):
        """Publish status message"""
        msg = String()
        msg.data = status_msg
        self.status_pub.publish(msg)
        
        # Also publish connection status
        self.publish_connection_status()
    
    def publish_connection_status(self):
        """Publish connection status periodically"""
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

