import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from mavros_msgs.srv import ParamSetV2
from rcl_interfaces.msg import ParameterValue, ParameterType
import time
import threading
from mavros_msgs.srv import ParamPull

SITL_VISION_PARAMS = {
    'EKF2_EV_CTRL': (ParameterType.PARAMETER_INTEGER, 15),          # External Vision: pos + vel + yaw
    'EKF2_HGT_REF': (ParameterType.PARAMETER_INTEGER, 3),           # Height reference = Vision
    'EKF2_GPS_CTRL': (ParameterType.PARAMETER_INTEGER, 0),          # Disable GPS
    'EKF2_EV_DELAY': (ParameterType.PARAMETER_DOUBLE, 5.0),         # Vision delay compensation (ms)
    'CBRK_SUPPLY_CHK': (ParameterType.PARAMETER_INTEGER, 894281),   # Bypass power supply check
    'COM_LOW_BAT_ACT': (ParameterType.PARAMETER_INTEGER, 0),        # No action on low battery
    'NAV_RCL_ACT': (ParameterType.PARAMETER_INTEGER, 0),            # No action on RC loss
    'COM_RCL_EXCEPT': (ParameterType.PARAMETER_INTEGER, 4),         # Allow OFFBOARD without RC
    'NAV_DLL_ACT': (ParameterType.PARAMETER_INTEGER, 0),            # No action on data link loss
    'GF_ACTION': (ParameterType.PARAMETER_INTEGER, 0),              # Disable geofence action
    'COM_DISARM_PRFLT': (ParameterType.PARAMETER_DOUBLE, -1.0),     # Never auto-disarm pre-flight
    'COM_OF_LOSS_T': (ParameterType.PARAMETER_DOUBLE, 5.0),         # OFFBOARD loss timeout
    'BAT_LOW_THR': (ParameterType.PARAMETER_DOUBLE, 0.05),          # 5% low threshold
    'BAT_CRIT_THR': (ParameterType.PARAMETER_DOUBLE, 0.03),         # 3% critical
    'BAT_EMERGEN_THR': (ParameterType.PARAMETER_DOUBLE, 0.01),      # 1% emergency
}

REAL_FLIGHT_PARAMS = {
    'EKF2_EV_CTRL': (ParameterType.PARAMETER_INTEGER, 15),
    'EKF2_HGT_REF': (ParameterType.PARAMETER_INTEGER, 3),
    'EKF2_GPS_CTRL': (ParameterType.PARAMETER_INTEGER, 0),
    'EKF2_EV_DELAY': (ParameterType.PARAMETER_DOUBLE, 5.0),
    'CBRK_SUPPLY_CHK': (ParameterType.PARAMETER_INTEGER, 0),
    'COM_LOW_BAT_ACT': (ParameterType.PARAMETER_INTEGER, 2),    # Return on low battery
    'NAV_RCL_ACT': (ParameterType.PARAMETER_INTEGER, 2),        # Return on RC loss
    'COM_RCL_EXCEPT': (ParameterType.PARAMETER_INTEGER, 4),     # Allow OFFBOARD without RC
    'NAV_DLL_ACT': (ParameterType.PARAMETER_INTEGER, 2),        # Return on data link loss
    'GF_ACTION': (ParameterType.PARAMETER_INTEGER, 2),          # Return on geofence violation
    'COM_DISARM_PRFLT': (ParameterType.PARAMETER_DOUBLE, 10.0), # Auto-disarm after 10s pre-flight
    'COM_OF_LOSS_T': (ParameterType.PARAMETER_DOUBLE, 3.0),     # OFFBOARD loss timeout (shorter)
    'BAT_LOW_THR': (ParameterType.PARAMETER_DOUBLE, 0.15),      # 15% low warning
    'BAT_CRIT_THR': (ParameterType.PARAMETER_DOUBLE, 0.10),     # 10% critical
    'BAT_EMERGEN_THR': (ParameterType.PARAMETER_DOUBLE, 0.05),  # 5% emergency
}

class Px4ParamSetup(Node):
    def __init__(self):
        super().__init__('px4_param_setup')

        self.declare_parameter('retry_interval', 2.0)       # seconds between retries
        self.declare_parameter('max_retries', 30)           # max attempts to reach param service
        self.declare_parameter('auto_shutdown', True)       # shutdown after params are set
        self.declare_parameter('deployment_mode', 'sitl')   # 'sitl' or 'real'
        self.retry_interval = self.get_parameter('retry_interval').value
        self.max_retries = self.get_parameter('max_retries').value
        self.auto_shutdown = self.get_parameter('auto_shutdown').value
        self.deployment_mode = self.get_parameter('deployment_mode').value

        if self.deployment_mode == 'real':
            self.get_logger().warn('═' * 50)
            self.get_logger().warn('  DEPLOYMENT MODE: REAL FLIGHT')
            self.get_logger().warn('  Failsafes ENABLED — safe params will be used')
            self.get_logger().warn('═' * 50)
            self._params_to_set = REAL_FLIGHT_PARAMS
        else:
            self.get_logger().info('Deployment mode: SITL (failsafes disabled for testing)')
            self._params_to_set = SITL_VISION_PARAMS
        self.service_cb_group = MutuallyExclusiveCallbackGroup()

        self.param_client_v2 = self.create_client(
            ParamSetV2, '/mavros/param/set',
            callback_group=self.service_cb_group
        )

        self.param_pull_client = self.create_client(
            ParamPull, '/mavros/param/pull',
            callback_group=self.service_cb_group
        )

        self.retry_count = 0
        self.params_set = 0
        self.params_failed = 0
        self.get_logger().info('PX4 Param Setup — waiting for MAVROS param service...')
        self.setup_timer = self.create_timer(self.retry_interval, self.attempt_setup)

    def attempt_setup(self):
        """Try to connect and set all params."""
        self.retry_count += 1

        if self.retry_count > self.max_retries:
            self.get_logger().error(
                f'Failed to connect to MAVROS param service after {self.max_retries} attempts. '
                f'Make sure MAVROS is running.'
            )
            self.setup_timer.cancel()
            if self.auto_shutdown:
                raise SystemExit(1)
            return

        if not self.param_client_v2.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                f'Waiting for /mavros/param/set service... '
                f'(attempt {self.retry_count}/{self.max_retries})',
            )
            return

        self.get_logger().info('MAVROS param service connected! Setting params...')
        self.setup_timer.cancel()

        thread = threading.Thread(target=self._set_all_params, daemon=True)
        thread.start()

    def _set_all_params(self):
        self._wait_for_param_download()

        for param_name, (param_type, param_value) in self._params_to_set.items():
            self._set_param_with_retry(param_name, param_type, param_value, max_retries=3)

        self.get_logger().info('')
        self.get_logger().info('═' * 50)
        self.get_logger().info(f'PX4 Param Setup complete!')
        self.get_logger().info(f'  ✅ Set: {self.params_set}/{len(self._params_to_set)}')
        if self.params_failed > 0:
            self.get_logger().warn(f'  ❌ Failed: {self.params_failed}/{len(self._params_to_set)}')
        self.get_logger().info('═' * 50)
        self.get_logger().info('')

        if self.auto_shutdown:
            self.get_logger().info('Auto-shutdown enabled. Node will exit.')
            self.create_timer(1.0, lambda: rclpy.shutdown())

    def _wait_for_param_download(self):
        if not self.param_pull_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('ParamPull service not available, proceeding anyway...')
            time.sleep(5.0)  # Fallback: just wait
            return

        self.get_logger().info('Requesting MAVROS param pull (sync with PX4)...')
        for attempt in range(5):
            req = ParamPull.Request()
            req.force_pull = False
            future = self.param_pull_client.call_async(req)

            deadline = time.time() + 15.0
            while not future.done() and time.time() < deadline:
                time.sleep(0.1)

            if future.done() and future.result() is not None:
                result = future.result()
                if result.success:
                    self.get_logger().info(
                        f'MAVROS param sync complete: {result.param_received} params received'
                    )
                    return
                else:
                    self.get_logger().warn(
                        f'Param pull attempt {attempt+1}/5: not ready yet '
                        f'({result.param_received} params), retrying...'
                    )
            else:
                self.get_logger().warn(
                    f'Param pull attempt {attempt+1}/5: timeout, retrying...'
                )
            time.sleep(3.0)

        self.get_logger().warn('Param download sync failed after 5 attempts, proceeding anyway...')

    def _set_param_with_retry(self, name, param_type, value, max_retries=3):
        """Set a param with retry logic."""
        for attempt in range(max_retries):
            success = self._set_param(name, param_type, value)
            if success:
                return
            if attempt < max_retries - 1:
                time.sleep(1.0)  # Wait before retry
        self.get_logger().warn(f'  ❌ {name} = {value} (failed after {max_retries} attempts)')

    def _set_param(self, name: str, param_type: int, value):
        req = ParamSetV2.Request()
        req.param_id = name
        req.value = ParameterValue()
        req.value.type = param_type

        if param_type == ParameterType.PARAMETER_INTEGER:
            req.value.integer_value = int(value)
        elif param_type == ParameterType.PARAMETER_DOUBLE:
            req.value.double_value = float(value)

        try:
            future = self.param_client_v2.call_async(req)
            deadline = time.time() + 10.0
            while not future.done() and time.time() < deadline:
                time.sleep(0.05)

            if not future.done():
                self.get_logger().warn(f'  ⏱ {name} = {value} (timeout)')
                self.params_failed += 1
                return False

            result = future.result()
            if result is not None and result.success:
                self.get_logger().info(f'  ✅ {name} = {value}')
                self.params_set += 1
                return True
            else:
                self.get_logger().debug(f'  ⚠ {name} = {value} (rejected, will retry)')
                return False
        except Exception as e:
            self.get_logger().warn(f'  ❌ {name}: {e}')
            self.params_failed += 1
            return False

def main(args=None):
    rclpy.init(args=args)
    node = Px4ParamSetup()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
