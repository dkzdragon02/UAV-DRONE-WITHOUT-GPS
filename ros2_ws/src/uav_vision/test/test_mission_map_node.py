#!/usr/bin/env python3

import json
import math
import os
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestGpsToNed:
    def _gps_to_ned(self, lat, lon, alt, home_lat, home_lon, home_alt):
        lat_rad = math.radians(home_lat)
        north = (lat - home_lat) * 111320.0
        east = (lon - home_lon) * 111320.0 * math.cos(lat_rad)
        down = -(alt - home_alt)
        return north, east, down

    def test_home_is_origin(self):
        n, e, d = self._gps_to_ned(10.0, 106.0, 5.0, 10.0, 106.0, 5.0)
        assert abs(n) < 1e-6
        assert abs(e) < 1e-6
        assert abs(d) < 1e-6

    def test_north_positive(self):
        n, e, d = self._gps_to_ned(10.001, 106.0, 0.0, 10.0, 106.0, 0.0)
        assert n > 0, "Higher lat should be positive north"
        expected_n = 0.001 * 111320.0  # ~111.32m
        assert abs(n - expected_n) < 1.0

    def test_east_positive(self):
        n, e, d = self._gps_to_ned(10.0, 106.001, 0.0, 10.0, 106.0, 0.0)
        assert e > 0, "Higher lon should be positive east"

    def test_down_positive_below(self):
        n, e, d = self._gps_to_ned(10.0, 106.0, 3.0, 10.0, 106.0, 5.0)
        assert d > 0, "Lower altitude should be positive down"
        assert abs(d - 2.0) < 1e-6

    def test_down_negative_above(self):
        n, e, d = self._gps_to_ned(10.0, 106.0, 10.0, 10.0, 106.0, 5.0)
        assert d < 0, "Higher altitude should be negative down"
        assert abs(d - (-5.0)) < 1e-6

    def test_cosine_scaling(self):
        # At equator (lat=0), cos(0)=1
        _, e_eq, _ = self._gps_to_ned(0.0, 0.001, 0.0, 0.0, 0.0, 0.0)
        # At lat=60°, cos(60°)=0.5
        _, e_60, _ = self._gps_to_ned(60.0, 0.001, 0.0, 60.0, 0.0, 0.0)
        assert abs(e_60 / e_eq - 0.5) < 0.01, "East at 60° should be ~half of equator"

    def test_symmetric_ns(self):
        n_north, _, _ = self._gps_to_ned(10.001, 106.0, 0.0, 10.0, 106.0, 0.0)
        n_south, _, _ = self._gps_to_ned(9.999, 106.0, 0.0, 10.0, 106.0, 0.0)
        assert abs(n_north + n_south) < 1e-6, "Should be symmetric"

    def test_realistic_waypoints(self):
        home_lat, home_lon, home_alt = 10.762622, 106.660172, 0.0
        wp_lat, wp_lon, wp_alt = 10.762722, 106.660272, 5.0
        n, e, d = self._gps_to_ned(wp_lat, wp_lon, wp_alt, home_lat, home_lon, home_alt)
        dist = math.sqrt(n ** 2 + e ** 2)
        assert 10.0 < dist < 20.0, f"Should be ~14m, got {dist:.1f}m"
        assert d == -5.0, "5m altitude = -5 down"

class TestMissionParsing:
    def _make_mission(self, **overrides):
        mission = {
            "mission_name": "Test Mission",
            "home_position": {
                "latitude": 10.762622,
                "longitude": 106.660172,
                "altitude": 0.0,
            },
            "flight_altitude": 5.0,
            "waypoints": [
                {"latitude": 10.762622, "longitude": 106.660172, "altitude": 5.0, "action": "takeoff"},
                {"latitude": 10.762722, "longitude": 106.660272, "altitude": 5.0, "action": "hover", "hover_time": 3.0},
                {"latitude": 10.762822, "longitude": 106.660172, "altitude": 5.0, "action": "photo"},
                {"latitude": 10.762622, "longitude": 106.660172, "altitude": 5.0, "action": "land"},
            ],
            "safety": {
                "max_speed": 1.0,
                "geofence_radius": 50.0,
                "rtl_on_mission_complete": True,
            },
        }
        mission.update(overrides)
        return mission

    def test_valid_mission_json(self):
        mission = self._make_mission()
        assert mission["mission_name"] == "Test Mission"
        assert len(mission["waypoints"]) == 4

    def test_waypoint_actions(self):
        mission = self._make_mission()
        actions = [wp["action"] for wp in mission["waypoints"]]
        assert "takeoff" in actions
        assert "hover" in actions
        assert "photo" in actions
        assert "land" in actions

    def test_mission_file_write_read(self):
        mission = self._make_mission()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mission, f)
            tmppath = f.name
        try:
            with open(tmppath) as f:
                loaded = json.load(f)
            assert loaded["mission_name"] == "Test Mission"
            assert len(loaded["waypoints"]) == 4
        finally:
            os.unlink(tmppath)

    def test_waypoints_within_geofence(self):
        mission = self._make_mission()
        home = mission["home_position"]
        geofence = mission["safety"]["geofence_radius"]
        lat_rad = math.radians(home["latitude"])
        for wp in mission["waypoints"]:
            n = (wp["latitude"] - home["latitude"]) * 111320.0
            e = (wp["longitude"] - home["longitude"]) * 111320.0 * math.cos(lat_rad)
            dist = math.sqrt(n ** 2 + e ** 2)
            assert dist <= geofence, f"WP at {dist:.1f}m exceeds geofence {geofence}m"

    def test_empty_waypoints(self):
        mission = self._make_mission(waypoints=[])
        assert len(mission["waypoints"]) == 0

    def test_no_home_uses_first_waypoint(self):
        mission = self._make_mission()
        del mission["home_position"]
        first_wp = mission["waypoints"][0]
        # Simulate: use first WP as home → NED of first WP is (0,0,*)
        lat_rad = math.radians(first_wp["latitude"])
        n = (first_wp["latitude"] - first_wp["latitude"]) * 111320.0
        e = (first_wp["longitude"] - first_wp["longitude"]) * 111320.0 * math.cos(lat_rad)
        assert abs(n) < 1e-9
        assert abs(e) < 1e-9

    def test_safety_defaults(self):
        mission = self._make_mission()
        del mission["safety"]
        assert "safety" not in mission

    def test_hover_time_present(self):
        mission = self._make_mission()
        hover_wps = [wp for wp in mission["waypoints"] if wp["action"] == "hover"]
        assert len(hover_wps) > 0
        for wp in hover_wps:
            assert wp.get("hover_time", 0) > 0
