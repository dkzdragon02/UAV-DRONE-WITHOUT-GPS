import numpy as np
from typing import List, Tuple
import math

class CoveragePlanner:
    def __init__(self):
        self.coverage_patterns = ['lawnmower', 'spiral', 'zigzag']
    
    def plan_lawnmower(self, bounds: Tuple[float, float, float, float],
                      altitude: float, spacing: float) -> List[List[float]]:
        x_min, y_min, x_max, y_max = bounds
        waypoints = []
        y = y_min
        direction = 1  # 1 for right, -1 for left
        
        while y <= y_max:
            if direction == 1:
                waypoints.append([x_min, y, altitude])
                waypoints.append([x_max, y, altitude])
            else:
                waypoints.append([x_max, y, altitude])
                waypoints.append([x_min, y, altitude])
            
            y += spacing
            direction *= -1
        
        return waypoints
    
    def plan_spiral(self, center: Tuple[float, float], 
                   max_radius: float, altitude: float,
                   spacing: float) -> List[List[float]]:
        waypoints = []
        cx, cy = center
        
        radius = spacing
        angle = 0.0
        
        while radius <= max_radius:
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            waypoints.append([x, y, altitude])
            angle += spacing / radius
            if angle >= 2 * math.pi:
                angle = 0.0
                radius += spacing
        
        return waypoints
    
    def plan_zigzag(self, start: Tuple[float, float],
                   end: Tuple[float, float],
                   altitude: float, spacing: float) -> List[List[float]]:
        waypoints = []
        sx, sy = start
        ex, ey = end
        dx = ex - sx
        dy = ey - sy
        length = math.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            return [[sx, sy, altitude]]
        dx /= length
        dy /= length
        perp_x = -dy
        perp_y = dx
        num_passes = int(length / spacing) + 1
        
        for i in range(num_passes + 1):
            t = i / num_passes if num_passes > 0 else 1.0
            x = sx + t * (ex - sx)
            y = sy + t * (ey - sy)
            offset = spacing * (i % 2) * 0.5                            # Alternate sides
            x += perp_x * offset
            y += perp_y * offset     
            waypoints.append([x, y, altitude])
        return waypoints
    
    def plan_rectangle(self, center: Tuple[float, float],
                      width: float, height: float,
                      altitude: float, spacing: float) -> List[List[float]]:
        cx, cy = center
        x_min = cx - width / 2
        x_max = cx + width / 2
        y_min = cy - height / 2
        y_max = cy + height / 2
        
        return self.plan_lawnmower((x_min, y_min, x_max, y_max), altitude, spacing)
    
    def plan_circle(self, center: Tuple[float, float],
                   radius: float, altitude: float,
                   spacing: float) -> List[List[float]]:
        return self.plan_spiral(center, radius, altitude, spacing)

