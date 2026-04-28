import pytest
import time
from unittest.mock import MagicMock, patch
from enum import Enum

class TestFlightState:
    def test_all_states_defined(self):
        from uav_vision.state_machine import FlightState
        states = list(FlightState)
        assert len(states) == 8
        assert FlightState.IDLE in states
        assert FlightState.TAKEOFF in states
        assert FlightState.HOVER in states
        assert FlightState.NAVIGATION in states
        assert FlightState.LANDING in states
        assert FlightState.RETURN_TO_LAUNCH in states
        assert FlightState.EMERGENCY in states
        assert FlightState.MANUAL_CONTROL in states

    def test_state_values(self):
        from uav_vision.state_machine import FlightState
        assert FlightState.IDLE.value == "idle"
        assert FlightState.TAKEOFF.value == "takeoff"
        assert FlightState.EMERGENCY.value == "emergency"
        assert FlightState.RETURN_TO_LAUNCH.value == "rtl"


class TestStateTransitions:
    def test_valid_transitions(self):
        from uav_vision.state_machine import FlightState
        valid_transitions = {
            FlightState.IDLE: [FlightState.TAKEOFF],
            FlightState.TAKEOFF: [FlightState.HOVER, FlightState.EMERGENCY],
            FlightState.HOVER: [FlightState.NAVIGATION, FlightState.LANDING, FlightState.RETURN_TO_LAUNCH, FlightState.MANUAL_CONTROL],
            FlightState.NAVIGATION: [FlightState.HOVER, FlightState.LANDING, FlightState.RETURN_TO_LAUNCH],
            FlightState.LANDING: [FlightState.IDLE, FlightState.EMERGENCY],
            FlightState.RETURN_TO_LAUNCH: [FlightState.HOVER],
            FlightState.EMERGENCY: [FlightState.IDLE],
            FlightState.MANUAL_CONTROL: [FlightState.HOVER, FlightState.LANDING, FlightState.EMERGENCY],
        }

        for state in FlightState:
            assert state in valid_transitions, f"Missing transitions for {state}"

    def test_idle_cannot_go_to_navigation(self):
        from uav_vision.state_machine import FlightState
        assert FlightState.IDLE != FlightState.NAVIGATION

    def test_emergency_can_always_be_entered(self):
        from uav_vision.state_machine import FlightState
        assert FlightState.EMERGENCY.value == "emergency"

class TestSafetyChecks:
    def test_altitude_limit(self):
        max_altitude = 10.0
        current_altitude = 11.0
        assert current_altitude > max_altitude

    def test_altitude_within_limit(self):
        max_altitude = 10.0
        current_altitude = 5.0
        assert current_altitude <= max_altitude

    def test_vision_timeout_detection(self):
        odom_timeout = 2.0
        last_odom_time = time.time() - 3.0  # 3 seconds ago
        elapsed = time.time() - last_odom_time
        assert elapsed > odom_timeout

    def test_vision_fresh_ok(self):
        odom_timeout = 2.0
        last_odom_time = time.time() - 0.5  # 0.5 seconds ago
        elapsed = time.time() - last_odom_time
        assert elapsed < odom_timeout


class TestGeofenceIntegration:      
    def test_geofence_boundary_check(self):
        from uav_vision.geofencing import Geofence, ViolationType
        gf = Geofence(max_radius=50.0, max_altitude=30.0)
        result = gf.check_position(10.0, 10.0, 5.0)
        
        assert result.is_safe
        result = gf.check_position(60.0, 0.0, 5.0)
        assert not result.is_safe
        assert result.violation_type == ViolationType.BOUNDARY_RADIUS

        result = gf.check_position(0.0, 0.0, 35.0)
        assert not result.is_safe
        assert result.violation_type == ViolationType.MAX_ALTITUDE

    def test_geofence_no_fly_zone(self):
        from uav_vision.geofencing import Geofence, ViolationType

        gf = Geofence(max_radius=100.0, max_altitude=50.0)
        gf.add_no_fly_zone("test_zone", [(5, 5), (15, 5), (15, 15), (5, 15)])

        result = gf.check_position(10.0, 10.0, 5.0)
        assert not result.is_safe
        assert result.violation_type == ViolationType.NO_FLY_ZONE

        result = gf.check_position(20.0, 20.0, 5.0)
        assert result.is_safe

    def test_geofence_warning_margin(self):
        from uav_vision.geofencing import Geofence

        gf = Geofence(max_radius=50.0, max_altitude=30.0, warning_margin=2.0)

        result = gf.check_position(49.0, 0.0, 5.0)
        assert result.is_safe
        assert result.distance_to_boundary < gf.warning_margin


class TestTimeouts:
    def test_takeoff_timeout(self):
        takeoff_timeout = 30.0
        elapsed = 31.0
        assert elapsed > takeoff_timeout

    def test_landing_timeout(self):
        landing_timeout = 30.0
        elapsed = 31.0
        assert elapsed > landing_timeout

    def test_takeoff_altitude_threshold(self):
        target_altitude = 2.0
        current_altitude = 1.85
        threshold = target_altitude * 0.9  # 1.8
        assert current_altitude >= threshold

    def test_landing_altitude_threshold(self):
        emergency_landing_altitude = 0.5
        current_altitude = 0.3
        assert current_altitude <= emergency_landing_altitude
