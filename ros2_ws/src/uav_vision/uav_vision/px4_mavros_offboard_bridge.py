#!/usr/bin/env python3
"""
PX4 MAVROS Offboard Bridge

- Nhận lệnh vận tốc từ controller nội bộ (ví dụ PX4Controller)
- Gửi setpoint velocity cho PX4 qua MAVROS
- Tự động ARM + chuyển sang OFFBOARD khi đã có kết nối

Thiết kế tối giản để chạy trong SITL:
- Input:  geometry_msgs/msg/TwistStamped trên topic cấu hình (mặc định: /uav/px4_controller/velocity)
- Output: geometry_msgs/msg/TwistStamped -> /mavros/setpoint_velocity/cmd_vel
"""

import time
from typing import Optional

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode


class Px4MavrosOffboardBridge(Node):
    """Bridge giữa controller nội bộ và PX4 thông qua MAVROS (OFFBOARD)"""

    def __init__(self) -> None:
        super().__init__("px4_mavros_offboard_bridge")

        # Parameters
        self.declare_parameter("velocity_input_topic", "/uav/px4_controller/velocity")
        self.declare_parameter("mavros_velocity_topic", "/mavros/setpoint_velocity/cmd_vel")
        self.declare_parameter("offboard_mode", "OFFBOARD")
        self.declare_parameter("auto_arm", True)
        self.declare_parameter("auto_offboard", True)
        self.declare_parameter("setpoint_rate_hz", 20.0)

        velocity_input_topic = self.get_parameter("velocity_input_topic").value
        mavros_velocity_topic = self.get_parameter("mavros_velocity_topic").value
        self.offboard_mode = self.get_parameter("offboard_mode").value
        self.auto_arm = bool(self.get_parameter("auto_arm").value)
        self.auto_offboard = bool(self.get_parameter("auto_offboard").value)
        self.setpoint_dt = 1.0 / float(self.get_parameter("setpoint_rate_hz").value)

        # State
        self.current_state: Optional[State] = None
        self.last_cmd: Optional[TwistStamped] = None
        self.last_state_time: float = 0.0

        # Subscribers
        self.state_sub = self.create_subscription(
            State,
            "/mavros/state",
            self.state_callback,
            10,
        )
        self.velocity_sub = self.create_subscription(
            TwistStamped,
            velocity_input_topic,
            self.velocity_callback,
            10,
        )

        # Publishers
        self.velocity_pub = self.create_publisher(
            TwistStamped,
            mavros_velocity_topic,
            10,
        )

        # Service clients
        self.arming_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.set_mode_client = self.create_client(SetMode, "/mavros/set_mode")

        # Timers
        self.setpoint_timer = self.create_timer(self.setpoint_dt, self.publish_setpoint)
        self.supervisor_timer = self.create_timer(1.0, self.offboard_supervisor)

        self.get_logger().info(
            f"PX4 MAVROS Offboard Bridge started. "
            f"Input: {velocity_input_topic}, Output: {mavros_velocity_topic}"
        )

    # --------------------------------------------------------------------- #
    # Callbacks
    # --------------------------------------------------------------------- #

    def state_callback(self, msg: State) -> None:
        self.current_state = msg
        self.last_state_time = time.time()

    def velocity_callback(self, msg: TwistStamped) -> None:
        """Lưu lệnh vận tốc cuối cùng từ controller nội bộ."""
        self.last_cmd = msg

    # --------------------------------------------------------------------- #
    # Core logic
    # --------------------------------------------------------------------- #

    def publish_setpoint(self) -> None:
        """
        Gửi setpoint velocity tới PX4 ở tần số cố định.

        PX4 yêu cầu stream OFFBOARD setpoints liên tục (>= 2Hz),
        nên ngay cả khi không có lệnh mới, ta giữ lại lệnh cuối.
        """
        if self.last_cmd is None:
            # Nếu chưa có lệnh nào, publish zero để giữ kết nối
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.twist.linear.x = 0.0
            msg.twist.linear.y = 0.0
            msg.twist.linear.z = 0.0
        else:
            msg = self.last_cmd
            msg.header.stamp = self.get_clock().now().to_msg()

        self.velocity_pub.publish(msg)

    def offboard_supervisor(self) -> None:
        """Định kỳ kiểm tra trạng thái PX4 và tự động ARM / OFFBOARD nếu cần."""
        if self.current_state is None:
            self.get_logger().warn("Waiting for /mavros/state ...", throttle_duration_sec=5.0)
            return

        if not self.current_state.connected:
            self.get_logger().warn("PX4 not connected (via MAVROS)", throttle_duration_sec=5.0)
            return

        # Arm if needed
        if self.auto_arm and not self.current_state.armed:
            self.try_arm()

        # Switch to OFFBOARD if needed
        if self.auto_offboard and self.current_state.mode != self.offboard_mode:
            self.try_set_mode(self.offboard_mode)

    # --------------------------------------------------------------------- #
    # Service helpers
    # --------------------------------------------------------------------- #

    def try_arm(self) -> None:
        if not self.arming_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn("Arming service not available")
            return

        req = CommandBool.Request()
        req.value = True

        self.get_logger().info("Sending ARM command ...")
        future = self.arming_client.call_async(req)

        def _done_cb(fut):
            try:
                resp = fut.result()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"Arm call failed: {exc}")
                return

            if resp.success:
                self.get_logger().info("ARM succeeded")
            else:
                self.get_logger().warn(f"ARM failed, result: {resp.result}")

        future.add_done_callback(_done_cb)

    def try_set_mode(self, mode: str) -> None:
        if not self.set_mode_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn("SetMode service not available")
            return

        req = SetMode.Request()
        req.custom_mode = mode

        self.get_logger().info(f"Setting mode to {mode} ...")
        future = self.set_mode_client.call_async(req)

        def _done_cb(fut):
            try:
                resp = fut.result()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"SetMode call failed: {exc}")
                return

            if resp.mode_sent:
                self.get_logger().info(f"Mode {mode} set request accepted")
            else:
                self.get_logger().warn(f"Failed to set mode {mode}")

        future.add_done_callback(_done_cb)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Px4MavrosOffboardBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


