#!/usr/bin/env python3
"""
Mission Planner
High-level mission planning and waypoint management
"""

import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time


class MissionState(Enum):
    """Mission state"""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Waypoint:
    """Waypoint definition"""
    position: List[float]  # [x, y, z]
    tolerance: float = 0.5  # meters
    action: Optional[str] = None  # Optional action at waypoint
    parameters: Dict = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class Mission:
    """Mission definition"""
    waypoints: List[Waypoint]
    mission_id: str
    priority: int = 0
    timeout: Optional[float] = None
    on_complete: Optional[Callable] = None
    on_fail: Optional[Callable] = None


class MissionPlanner:
    """
    Mission planner for managing complex missions.
    
    Features:
    - Waypoint management
    - Mission state tracking
    - Progress monitoring
    - Mission sequencing
    """
    
    def __init__(self):
        """Initialize mission planner"""
        self.current_mission: Optional[Mission] = None
        self.mission_queue: List[Mission] = []
        self.state = MissionState.IDLE
        self.current_waypoint_index = 0
        self.mission_start_time: Optional[float] = None
        self.position_callback: Optional[Callable] = None
    
    def add_mission(self, mission: Mission):
        """
        Add mission to queue.
        
        Args:
            mission: Mission to add
        """
        self.mission_queue.append(mission)
        self.mission_queue.sort(key=lambda m: m.priority, reverse=True)
    
    def start_mission(self, mission_id: Optional[str] = None) -> bool:
        """
        Start a mission.
        
        Args:
            mission_id: Mission ID (None for next in queue)
            
        Returns:
            True if started, False otherwise
        """
        if self.state != MissionState.IDLE:
            return False
        
        # Find mission
        if mission_id:
            mission = next(
                (m for m in self.mission_queue if m.mission_id == mission_id),
                None
            )
        else:
            if not self.mission_queue:
                return False
            mission = self.mission_queue.pop(0)
        
        if not mission:
            return False
        
        self.current_mission = mission
        self.state = MissionState.EXECUTING
        self.current_waypoint_index = 0
        self.mission_start_time = time.time()
        
        return True
    
    def update_position(self, position: List[float]):
        """
        Update current position and check waypoint progress.
        
        Args:
            position: Current position [x, y, z]
        """
        if self.state != MissionState.EXECUTING:
            return
        
        if not self.current_mission:
            return
        
        # Check if current waypoint reached
        if self.current_waypoint_index < len(self.current_mission.waypoints):
            waypoint = self.current_mission.waypoints[self.current_waypoint_index]
            distance = np.linalg.norm(
                np.array(position) - np.array(waypoint.position)
            )
            
            if distance <= waypoint.tolerance:
                self._waypoint_reached()
        
        # Check timeout
        if self.current_mission.timeout:
            elapsed = time.time() - self.mission_start_time
            if elapsed > self.current_mission.timeout:
                self._mission_failed("Timeout")
    
    def _waypoint_reached(self):
        """Handle waypoint reached"""
        waypoint = self.current_mission.waypoints[self.current_waypoint_index]
        
        # Execute waypoint action if specified
        if waypoint.action:
            # Action execution would go here
            pass
        
        # Move to next waypoint
        self.current_waypoint_index += 1
        
        # Check if mission complete
        if self.current_waypoint_index >= len(self.current_mission.waypoints):
            self._mission_completed()
    
    def _mission_completed(self):
        """Handle mission completion"""
        self.state = MissionState.COMPLETED
        
        if self.current_mission and self.current_mission.on_complete:
            self.current_mission.on_complete()
        
        self._reset()
    
    def _mission_failed(self, reason: str):
        """Handle mission failure"""
        self.state = MissionState.FAILED
        
        if self.current_mission and self.current_mission.on_fail:
            self.current_mission.on_fail(reason)
        
        self._reset()
    
    def pause_mission(self):
        """Pause current mission"""
        if self.state == MissionState.EXECUTING:
            self.state = MissionState.PAUSED
    
    def resume_mission(self):
        """Resume paused mission"""
        if self.state == MissionState.PAUSED:
            self.state = MissionState.EXECUTING
    
    def cancel_mission(self):
        """Cancel current mission"""
        if self.state in [MissionState.EXECUTING, MissionState.PAUSED]:
            self.state = MissionState.CANCELLED
            self._reset()
    
    def _reset(self):
        """Reset mission state"""
        self.current_mission = None
        self.current_waypoint_index = 0
        self.mission_start_time = None
        self.state = MissionState.IDLE
    
    def get_current_waypoint(self) -> Optional[Waypoint]:
        """Get current waypoint"""
        if not self.current_mission:
            return None
        
        if self.current_waypoint_index < len(self.current_mission.waypoints):
            return self.current_mission.waypoints[self.current_waypoint_index]
        
        return None
    
    def get_progress(self) -> Dict:
        """
        Get mission progress.
        
        Returns:
            Progress dictionary
        """
        if not self.current_mission:
            return {
                'state': self.state.value,
                'progress': 0.0,
                'waypoints_completed': 0,
                'total_waypoints': 0
            }
        
        total = len(self.current_mission.waypoints)
        completed = self.current_waypoint_index
        progress = completed / total if total > 0 else 0.0
        
        return {
            'state': self.state.value,
            'progress': progress,
            'waypoints_completed': completed,
            'total_waypoints': total,
            'mission_id': self.current_mission.mission_id
        }
    
    def get_next_waypoint(self) -> Optional[Waypoint]:
        """Get next waypoint in queue"""
        if not self.mission_queue:
            return None
        
        return self.mission_queue[0].waypoints[0] if self.mission_queue[0].waypoints else None

