import json
import math
import os
from typing import Dict, List, Optional, Tuple
import rclpy
from rclpy.node import Node
from uav_vision.qos_profiles import mavros_qos as make_mavros_qos
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import String, Bool
import numpy as np

class MissionWaypoint:
    def __init__(self, latitude: float, longitude: float, altitude: float,
                 action: str = "flyover", hover_time: float = 0.0,
                 index: int = 0):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.action = action            # takeoff, hover, photo, land, flyover
        self.hover_time = hover_time
        self.index = index
        self.north = 0.0
        self.east = 0.0
        self.down = 0.0                 # NED convention: down = -altitude

    def __repr__(self):
        return (f"WP#{self.index}({self.latitude:.6f},{self.longitude:.6f},"
                f"alt={self.altitude:.1f},act={self.action}) "
                f"→ NED({self.north:.2f},{self.east:.2f},{self.down:.2f})")

class MissionMapNode(Node):
    def __init__(self) -> None:
        super().__init__("mission_map_node")

        self.declare_parameter("mission_file", "")
        self.declare_parameter("auto_start", False)
        self.declare_parameter("waypoint_reached_radius", 1.0)   # meters
        self.declare_parameter("publish_rate", 1.0)              # Hz
        self.declare_parameter("max_speed", 1.0)                 # m/s (safety)
        self.declare_parameter("geofence_radius", 50.0)          # meters
        self.mission_file = self.get_parameter("mission_file").get_parameter_value().string_value
        self.auto_start = bool(self.get_parameter("auto_start").value)
        self.waypoint_reached_radius = float(self.get_parameter("waypoint_reached_radius").value)
        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.max_speed = float(self.get_parameter("max_speed").value)
        self.geofence_radius = float(self.get_parameter("geofence_radius").value)
        self.mission_name: str = ""
        self.waypoints: List[MissionWaypoint] = []
        self.home_position: Optional[Dict] = None
        self.safety_config: Optional[Dict] = None
        self.flight_altitude: float = 5.0
        self.current_waypoint_index: int = 0
        self.mission_loaded: bool = False
        self.mission_active: bool = False
        self.mission_paused: bool = False
        self.mission_complete: bool = False
        self.flight_state: str = "idle"
        self.current_position: Optional[np.ndarray] = None  # [x, y, z] NED
        self.waypoints_pub = self.create_publisher(
            Path, "/uav/path_planner/waypoints", 10
        )
        self.status_pub = self.create_publisher(
            String, "/uav/mission_map/status", 10
        )
        self.progress_pub = self.create_publisher(
            String, "/uav/mission_map/progress", 10
        )

        self.state_command_pub = self.create_publisher(
            String, "/uav/state_machine/command", 10
        )
        self.camera_capture_pub = self.create_publisher(
            Bool, "/uav/camera/capture", 10
        )
        self._hover_timer = None


        self.state_sub = self.create_subscription(
            String, "/uav/state_machine/state",
            self.state_callback, 10
        )

        mavros_qos = make_mavros_qos()
        self.pose_sub = self.create_subscription(
            PoseStamped, "/mavros/local_position/pose",
            self.pose_callback, mavros_qos
        )


        self.load_mission_sub = self.create_subscription(
            String, "/uav/mission_map/load_mission",
            self.load_mission_callback, 10
        )

        self.command_sub = self.create_subscription(
            String, "/uav/mission_map/command",
            self.command_callback, 10
        )

        self.progress_timer = self.create_timer(
            1.0 / self.publish_rate, self.progress_tick
        )

        if self.mission_file:
            self.load_mission_from_file(self.mission_file)

        self.get_logger().info("MissionMapNode started")
        if self.mission_file:
            self.get_logger().info(f"  Mission file: {self.mission_file}")
        self.get_logger().info(f"  Auto-start: {self.auto_start}")
        self.get_logger().info(f"  WP reached radius: {self.waypoint_reached_radius}m")


    @staticmethod
    def gps_to_ned(lat: float, lon: float, alt: float,
                   home_lat: float, home_lon: float, home_alt: float
                   ) -> Tuple[float, float, float]:
        lat_rad = math.radians(home_lat)
        north = (lat - home_lat) * 111320.0
        east = (lon - home_lon) * 111320.0 * math.cos(lat_rad)
        down = -(alt - home_alt)  # NED: positive down

        return north, east, down

    def load_mission_from_file(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            self.get_logger().error(f"Mission file not found: {filepath}")
            self._publish_status("ERROR", f"File not found: {filepath}")
            return False

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            return self._parse_mission(data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Invalid JSON in mission file: {e}")
            self._publish_status("ERROR", f"Invalid JSON: {e}")
            return False
        except Exception as e:
            self.get_logger().error(f"Failed to load mission: {e}")
            self._publish_status("ERROR", str(e))
            return False

    def load_mission_from_json(self, json_str: str) -> bool:
        try:
            data = json.loads(json_str)
            return self._parse_mission(data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Invalid JSON string: {e}")
            self._publish_status("ERROR", f"Invalid JSON: {e}")
            return False

    def _parse_mission(self, data: dict) -> bool:
        self.mission_name = data.get("mission_name", "Unnamed Mission")
        self.flight_altitude = data.get("flight_altitude", 5.0)
        self.safety_config = data.get("safety", {})

        if self.safety_config and "max_speed" in self.safety_config:
            self.max_speed = self.safety_config["max_speed"]

        if self.safety_config and "geofence_radius" in self.safety_config:
            self.geofence_radius = self.safety_config["geofence_radius"]

        home = data.get("home_position", {})
        if not home:
            wps = data.get("waypoints", [])
            if wps:
                home = {
                    "latitude": wps[0].get("latitude", 0),
                    "longitude": wps[0].get("longitude", 0),
                    "altitude": wps[0].get("altitude", 0),
                }
            else:
                self.get_logger().error("No home position or waypoints in mission")
                self._publish_status("ERROR", "No waypoints")
                return False

        self.home_position = home
        home_lat = home["latitude"]
        home_lon = home["longitude"]
        home_alt = home.get("altitude", 0.0)

        raw_waypoints = data.get("waypoints", [])
        if not raw_waypoints:
            self.get_logger().error("No waypoints in mission file")
            self._publish_status("ERROR", "No waypoints")
            return False

        self.waypoints = []
        for i, wp in enumerate(raw_waypoints):
            mwp = MissionWaypoint(
                latitude=wp["latitude"],
                longitude=wp["longitude"],
                altitude=wp.get("altitude", self.flight_altitude),
                action=wp.get("action", "flyover"),
                hover_time=wp.get("hover_time", 0.0),
                index=i,
            )

            mwp.north, mwp.east, mwp.down = self.gps_to_ned(
                mwp.latitude, mwp.longitude, mwp.altitude,
                home_lat, home_lon, home_alt,
            )

            horizontal_dist = math.sqrt(mwp.north ** 2 + mwp.east ** 2)
            if horizontal_dist > self.geofence_radius:
                self.get_logger().warn(
                    f"WP#{i} ({horizontal_dist:.1f}m) exceeds geofence "
                    f"({self.geofence_radius:.0f}m) — skipping"
                )
                continue

            self.waypoints.append(mwp)
            self.get_logger().info(f"  {mwp}")

        if not self.waypoints:
            self.get_logger().error("All waypoints filtered by geofence!")
            self._publish_status("ERROR", "No valid waypoints after geofence filter")
            return False

        self.mission_loaded = True
        self.mission_complete = False
        self.current_waypoint_index = 0

        self.get_logger().info(
            f'Mission loaded: "{self.mission_name}" — '
            f'{len(self.waypoints)} waypoints, '
            f'flight alt={self.flight_altitude}m'
        )
        self._publish_status(
            "LOADED",
            f'Mission "{self.mission_name}" loaded: {len(self.waypoints)} waypoints'
        )

        self._publish_waypoints()

        if self.auto_start and self.flight_state != "idle":
            self.start_mission()

        return True

    def _publish_waypoints(self) -> None:
        if not self.waypoints:
            return

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"

        for wp in self.waypoints:
            pose = PoseStamped()
            pose.header.stamp = path_msg.header.stamp
            pose.header.frame_id = "map"
            pose.pose.position.x = wp.north
            pose.pose.position.y = wp.east
            pose.pose.position.z = -wp.down  # Convert NED down → ROS up
            path_msg.poses.append(pose)

        self.waypoints_pub.publish(path_msg)
        self.get_logger().info(
            f"Published {len(path_msg.poses)} NED waypoints to /uav/path_planner/waypoints"
        )

    def _publish_remaining_waypoints(self) -> None:
        if not self.waypoints or self.current_waypoint_index >= len(self.waypoints):
            return

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"

        for wp in self.waypoints[self.current_waypoint_index:]:
            pose = PoseStamped()
            pose.header.stamp = path_msg.header.stamp
            pose.header.frame_id = "map"
            pose.pose.position.x = wp.north
            pose.pose.position.y = wp.east
            pose.pose.position.z = -wp.down
            path_msg.poses.append(pose)

        self.waypoints_pub.publish(path_msg)

    def start_mission(self) -> None:
        if not self.mission_loaded:
            self.get_logger().warn("Cannot start — no mission loaded")
            self._publish_status("ERROR", "No mission loaded")
            return

        self.mission_active = True
        self.mission_paused = False
        self.mission_complete = False
        self.current_waypoint_index = 0
        self.get_logger().info(f'Mission "{self.mission_name}" STARTED')
        self._publish_status("STARTED", f'Mission "{self.mission_name}" started')
        self._publish_waypoints()
        self._handle_waypoint_action(self.waypoints[0])

    def pause_mission(self) -> None:
        if not self.mission_active:
            return
        self.mission_paused = True
        self.get_logger().info("⏸ Mission PAUSED")
        self._publish_status("PAUSED", "Mission paused")
        cmd = String()
        cmd.data = "hover"
        self.state_command_pub.publish(cmd)

    def resume_mission(self) -> None:
        if not self.mission_active or not self.mission_paused:
            return
        self.mission_paused = False
        self.get_logger().info("▶ Mission RESUMED")
        self._publish_status("RESUMED", "Mission resumed")
        self._publish_remaining_waypoints()
        
        cmd = String()
        cmd.data = "navigate"
        self.state_command_pub.publish(cmd)

    def abort_mission(self) -> None:
        self.mission_active = False
        self.mission_paused = False
        self.get_logger().warn("Mission ABORTED")
        self._publish_status("ABORTED", "Mission aborted")

        cmd = String()
        cmd.data = "hover"
        self.state_command_pub.publish(cmd)

    def state_callback(self, msg: String) -> None:
        self.flight_state = msg.data.lower().strip()

    def pose_callback(self, msg: PoseStamped) -> None:
        self.current_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])

    def load_mission_callback(self, msg: String) -> None:
        self.get_logger().info("Received mission via topic — loading...")
        self.load_mission_from_json(msg.data)

    def command_callback(self, msg: String) -> None:
        command = msg.data.lower().strip()
        self.get_logger().info(f'Mission command: "{command}"')

        if command == "start":
            self.start_mission()
        elif command == "pause":
            self.pause_mission()
        elif command == "resume":
            self.resume_mission()
        elif command == "abort":
            self.abort_mission()
        elif command == "reload":
            if self.mission_file:
                self.load_mission_from_file(self.mission_file)
            else:
                self.get_logger().warn("No mission file to reload")
        else:
            self.get_logger().warn(f'Unknown mission command: "{command}"')

    def progress_tick(self) -> None:
        """Periodic progress check and publishing."""
        if not self.mission_active or self.mission_paused or self.mission_complete:
            return

        if self.current_position is None or not self.waypoints:
            return

        if self.current_waypoint_index < len(self.waypoints):
            wp = self.waypoints[self.current_waypoint_index]
            wp_pos = np.array([wp.north, wp.east, -wp.down])  # NED→ROS
            dist = float(np.linalg.norm(self.current_position - wp_pos))

            progress_msg = String()
            progress_msg.data = json.dumps({
                "mission": self.mission_name,
                "waypoint": self.current_waypoint_index + 1,
                "total": len(self.waypoints),
                "action": wp.action,
                "distance_to_wp": round(dist, 2),
                "status": "active",
            })
            self.progress_pub.publish(progress_msg)

            if dist < self.waypoint_reached_radius:
                self.get_logger().info(
                    f"Waypoint {self.current_waypoint_index + 1}/{len(self.waypoints)} "
                    f"reached (dist={dist:.2f}m, action={wp.action})"
                )

                self._handle_waypoint_action(wp)

                self.current_waypoint_index += 1

                if self.current_waypoint_index >= len(self.waypoints):
                    self._mission_completed()
                else:
                    self._publish_remaining_waypoints()

    def _handle_waypoint_action(self, wp: MissionWaypoint) -> None:
        if wp.action == "takeoff":
            cmd = String()
            cmd.data = "takeoff"
            self.state_command_pub.publish(cmd)
            self.get_logger().info(f"WP#{wp.index}: Takeoff command sent")

        elif wp.action == "land":
            cmd = String()
            cmd.data = "land"
            self.state_command_pub.publish(cmd)
            self.get_logger().info(f"WP#{wp.index}: Land command sent")

        elif wp.action == "hover":
            self.get_logger().info(
                f"WP#{wp.index}: Hovering for {wp.hover_time}s"
            )

            if wp.hover_time > 0:
                self.mission_paused = True
                if self._hover_timer is not None:
                    self._hover_timer.cancel()
                    self.destroy_timer(self._hover_timer)
                self._hover_timer = self.create_timer(
                    wp.hover_time,
                    self._hover_complete_callback,
                )

        elif wp.action == "photo":
            self.get_logger().info(f"WP#{wp.index}: Capturing photo")
            capture_msg = Bool()
            capture_msg.data = True
            self.camera_capture_pub.publish(capture_msg)
            self.mission_paused = True
            if self._hover_timer is not None:
                self._hover_timer.cancel()
                self.destroy_timer(self._hover_timer)
            self._hover_timer = self.create_timer(
                1.5,
                self._hover_complete_callback,
            )

        elif wp.action == "flyover":
            pass  

        else:
            self.get_logger().info(f"WP#{wp.index}: Unknown action '{wp.action}'")

    def _hover_complete_callback(self) -> None:
        if self._hover_timer is not None:
            self._hover_timer.cancel()
            self.destroy_timer(self._hover_timer)
            self._hover_timer = None
        if self.mission_paused and self.mission_active:
            self.mission_paused = False
            self.get_logger().info("Hover complete — resuming mission")

    def _mission_completed(self) -> None:
        self.mission_complete = True
        self.mission_active = False
        self.get_logger().info(
            f'Mission "{self.mission_name}" COMPLETED — '
            f'all {len(self.waypoints)} waypoints reached'
        )
        self._publish_status("COMPLETED", f'Mission "{self.mission_name}" completed')

        if self.safety_config and self.safety_config.get("rtl_on_mission_complete", False):
            cmd = String()
            cmd.data = "rtl"
            self.state_command_pub.publish(cmd)
            self.get_logger().info("RTL commanded (rtl_on_mission_complete=true)")

    def _publish_status(self, status: str, message: str) -> None:
        msg = String()
        msg.data = json.dumps({
            "status": status,
            "message": message,
            "mission": self.mission_name,
            "waypoints_total": len(self.waypoints),
            "waypoint_current": self.current_waypoint_index,
        })
        self.status_pub.publish(msg)

    def get_mission_summary(self) -> dict:
        return {
            "name": self.mission_name,
            "loaded": self.mission_loaded,
            "active": self.mission_active,
            "paused": self.mission_paused,
            "complete": self.mission_complete,
            "waypoints_total": len(self.waypoints),
            "waypoint_current": self.current_waypoint_index,
            "flight_altitude": self.flight_altitude,
        }

def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
