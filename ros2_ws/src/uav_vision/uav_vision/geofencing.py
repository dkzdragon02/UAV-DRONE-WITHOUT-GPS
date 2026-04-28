import numpy as np
import math
import yaml
import logging
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

class ViolationType(Enum):
    NONE =              "none"
    MAX_ALTITUDE =      "max_altitude"
    MIN_ALTITUDE =      "min_altitude"
    BOUNDARY_RADIUS =   "boundary_radius"
    NO_FLY_ZONE =       "no_fly_zone"
    MAX_DISTANCE =      "max_distance"


@dataclass
class GeofenceResult:
    is_safe: bool
    violation_type: ViolationType = ViolationType.NONE
    distance_to_boundary: float = float('inf')
    message: str = ""

@dataclass
class NoFlyZone:
    name: str
    vertices: List[Tuple[float, float]]  # List of (x, y) vertices
    min_altitude: float = 0.0
    max_altitude: float = float('inf')
    active: bool = True

class Geofence:
    def __init__(self,
                 max_radius: float = 50.0,
                 max_altitude: float = 30.0,
                 min_altitude: float = 0.0,
                 origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 warning_margin: float = 2.0):
        self.max_radius = max_radius
        self.max_altitude = max_altitude
        self.min_altitude = min_altitude
        self.origin = np.array(origin)
        self.warning_margin = warning_margin
        self.no_fly_zones: List[NoFlyZone] = []
        self.total_checks = 0
        self.violations = 0
        self.warnings = 0
        
        logger.info(
            f"Geofence initialized: radius={max_radius}m, "
            f"alt=[{min_altitude}, {max_altitude}]m, "
            f"warning_margin={warning_margin}m"
        )

    def check_position(self, x: float, y: float, z: float) -> GeofenceResult:
        self.total_checks += 1
        if z > self.max_altitude:
            self.violations += 1
            return GeofenceResult(
                is_safe=False,
                violation_type=ViolationType.MAX_ALTITUDE,
                distance_to_boundary=z - self.max_altitude,
                message=f"Altitude {z:.1f}m exceeds max {self.max_altitude:.1f}m"
            )
        
        if z < self.min_altitude:
            self.violations += 1
            return GeofenceResult(
                is_safe=False,
                violation_type=ViolationType.MIN_ALTITUDE,
                distance_to_boundary=self.min_altitude - z,
                message=f"Altitude {z:.1f}m below min {self.min_altitude:.1f}m"
            )
        
        dx = x - self.origin[0]
        dy = y - self.origin[1]
        horizontal_distance = math.sqrt(dx * dx + dy * dy)
        distance_to_boundary = self.max_radius - horizontal_distance
        
        if horizontal_distance > self.max_radius:
            self.violations += 1
            return GeofenceResult(
                is_safe=False,
                violation_type=ViolationType.BOUNDARY_RADIUS,
                distance_to_boundary=-distance_to_boundary,  
                message=f"Distance {horizontal_distance:.1f}m exceeds boundary radius {self.max_radius:.1f}m"
            )
        
        for zone in self.no_fly_zones:
            if not zone.active:
                continue
            
            if z < zone.min_altitude or z > zone.max_altitude:
                continue
            
            if self._point_in_polygon(x, y, zone.vertices):
                self.violations += 1
                return GeofenceResult(
                    is_safe=False,
                    violation_type=ViolationType.NO_FLY_ZONE,
                    distance_to_boundary=0.0,
                    message=f"Inside no-fly zone: {zone.name}"
                )
        
        min_dist = min(
            distance_to_boundary,
            self.max_altitude - z
        )
        
        if min_dist < self.warning_margin:
            self.warnings += 1
        
        return GeofenceResult(
            is_safe=True,
            violation_type=ViolationType.NONE,
            distance_to_boundary=min_dist,
            message=""
        )

    def is_near_boundary(self, x: float, y: float, z: float) -> bool:
        result = self.check_position(x, y, z)
        return result.distance_to_boundary < self.warning_margin

    def add_no_fly_zone(self, name: str, vertices: List[Tuple[float, float]],
                        min_alt: float = 0.0, max_alt: float = float('inf')):
        zone = NoFlyZone(
            name=name,
            vertices=vertices,
            min_altitude=min_alt,
            max_altitude=max_alt
        )
        self.no_fly_zones.append(zone)
        logger.info(f"Added no-fly zone: {name} with {len(vertices)} vertices")

    def remove_no_fly_zone(self, name: str):
        self.no_fly_zones = [z for z in self.no_fly_zones if z.name != name]
        logger.info(f"Removed no-fly zone: {name}")

    def _point_in_polygon(self, px: float, py: float,
                          vertices: List[Tuple[float, float]]) -> bool:
        n = len(vertices)
        if n < 3:
            return False
        
        inside = False
        j = n - 1
        
        for i in range(n):
            xi, yi = vertices[i]
            xj, yj = vertices[j]
            
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            
            j = i
        
        return inside

    def load_from_config(self, config_path: str):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if 'geofence' not in config:
                logger.warning("No 'geofence' section in config file")
                return
            
            gc = config['geofence']
            
            if 'max_radius' in gc:
                self.max_radius = float(gc['max_radius'])
            if 'max_altitude' in gc:
                self.max_altitude = float(gc['max_altitude'])
            if 'min_altitude' in gc:
                self.min_altitude = float(gc['min_altitude'])
            if 'warning_margin' in gc:
                self.warning_margin = float(gc['warning_margin'])
            if 'origin' in gc:
                self.origin = np.array(gc['origin'], dtype=float)
            
            if 'no_fly_zones' in gc:
                for zone_config in gc['no_fly_zones']:
                    self.add_no_fly_zone(
                        name=zone_config.get('name', 'unnamed'),
                        vertices=[tuple(v) for v in zone_config.get('vertices', [])],
                        min_alt=zone_config.get('min_altitude', 0.0),
                        max_alt=zone_config.get('max_altitude', float('inf'))
                    )
            
            logger.info(f"Geofence config loaded from {config_path}")
            
        except FileNotFoundError:
            logger.warning(f"Geofence config file not found: {config_path}")
        except Exception as e:
            logger.error(f"Error loading geofence config: {e}")

    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_checks': self.total_checks,
            'violations': self.violations,
            'warnings': self.warnings,
            'no_fly_zones_count': len(self.no_fly_zones),
            'max_radius': self.max_radius,
            'max_altitude': self.max_altitude,
        }

    def update_boundary(self, max_radius: Optional[float] = None,
                       max_altitude: Optional[float] = None,
                       min_altitude: Optional[float] = None):
        if max_radius is not None:
            self.max_radius = max_radius
        if max_altitude is not None:
            self.max_altitude = max_altitude
        if min_altitude is not None:
            self.min_altitude = min_altitude
        logger.info(
            f"Geofence updated: radius={self.max_radius}m, "
            f"alt=[{self.min_altitude}, {self.max_altitude}]m"
        )
