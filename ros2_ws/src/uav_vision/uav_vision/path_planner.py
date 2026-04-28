import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Header
import numpy as np
import math
from typing import List, Tuple, Optional

class PathPlanner(Node):
    def __init__(self):
        super().__init__('path_planner')
        self.declare_parameter('map_topic', '/uav/slam/map')
        self.declare_parameter('pose_topic', '/uav/slam/pose')
        self.declare_parameter('path_topic', '/uav/path_planner/path')
        self.declare_parameter('waypoint_topic', '/uav/path_planner/waypoints')  
        self.declare_parameter('waypoint_tolerance', 0.5)   # meters
        self.declare_parameter('path_resolution', 0.1)      # meters
        self.declare_parameter('obstacle_inflation', 0.3)   # meters
        
        map_topic = self.get_parameter('map_topic').value
        pose_topic = self.get_parameter('pose_topic').value
        path_topic = self.get_parameter('path_topic').value
        waypoint_topic = self.get_parameter('waypoint_topic').value
        
        self.waypoint_tolerance = self.get_parameter('waypoint_tolerance').value
        self.path_resolution = self.get_parameter('path_resolution').value
        self.obstacle_inflation = self.get_parameter('obstacle_inflation').value
        self.current_pose = None
        self.occupancy_map = None
        self.map_info = None
        self.waypoints = []
        self.current_path = None
        self._inflated_map = None  # Precomputed boolean obstacle mask
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            map_topic,
            self.map_callback,
            10
        )
        
        self.pose_sub = self.create_subscription(
            PoseStamped,
            pose_topic,
            self.pose_callback,
            10
        )
        
        self.waypoint_sub = self.create_subscription(
            Path,
            waypoint_topic,
            self.waypoint_callback,
            10
        )
        
        self.path_pub = self.create_publisher(Path, path_topic, 10)
        self.planning_timer = self.create_timer(1.0, self.plan_path)
        self.get_logger().info('Path Planner started')
    
    def map_callback(self, msg):
        self.occupancy_map = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info
        self._precompute_inflated_map()
    
    def pose_callback(self, msg):
        self.current_pose = msg
    
    def waypoint_callback(self, msg):
        self.waypoints = []
        for pose_stamped in msg.poses:
            waypoint = [
                pose_stamped.pose.position.x,
                pose_stamped.pose.position.y,
                pose_stamped.pose.position.z
            ]
            self.waypoints.append(waypoint)
        
        self.get_logger().info(f'Received {len(self.waypoints)} waypoints')
        self.current_path = None  
    
    def plan_path(self):
        if not self.waypoints or self.current_pose is None:
            return
        
        if self.occupancy_map is None:
            self.plan_straight_line_path()
            return
        
        if self.current_path and len(self.current_path.poses) > 0:
            last_pose = self.current_path.poses[-1]
            distance = self.distance(
                self.current_pose.pose.position,
                last_pose.pose.position
            )
            
            if distance < self.waypoint_tolerance:
                if self.waypoints:
                    self.waypoints.pop(0)
                    self.current_path = None
        
        if self.waypoints and (self.current_path is None or len(self.current_path.poses) == 0):
            target = self.waypoints[0]
            self.plan_path_to_target(target)
    
    def plan_straight_line_path(self):
        if not self.waypoints or self.current_pose is None:
            return
        
        target = self.waypoints[0]
        start = self.current_pose.pose.position
    
        path = Path()
        path.header = Header()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "map"
        
        distance = self.distance(start, Point(x=float(target[0]), y=float(target[1]), z=float(target[2])))
        num_points = int(distance / self.path_resolution) + 1
        
        for i in range(num_points + 1):
            t = i / num_points if num_points > 0 else 1.0
            
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = start.x + t * (target[0] - start.x)
            pose.pose.position.y = start.y + t * (target[1] - start.y)
            pose.pose.position.z = start.z + t * (target[2] - start.z)
            pose.pose.orientation.w = 1.0
            
            path.poses.append(pose)
        
        self.current_path = path
        self.path_pub.publish(path)
    
    def plan_path_to_target(self, target: List[float]):
        if self.current_pose is None or self.occupancy_map is None:
            return
        
        start = self.current_pose.pose.position
        start_pos = [start.x, start.y]
        target_pos = target[:2]
        
        path_points = self.a_star_path(start_pos, target_pos)
        
        if not path_points:
            self.plan_straight_line_path()
            return
        
        path = Path()
        path.header = Header()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "map"
        
        for point in path_points:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = float(target[2]) if len(target) > 2 else 0.0
            pose.pose.orientation.w = 1.0
            
            path.poses.append(pose)
        
        self.current_path = path
        self.path_pub.publish(path)
    
    def a_star_path(self, start: List[float], goal: List[float]) -> List[List[float]]:
        if self.map_info is None:
            return []
        return self.a_star_improved(start, goal)
    
    def a_star_improved(self, start: List[float], goal: List[float]) -> List[List[float]]:
        if self.map_info is None or self.occupancy_map is None:
            return []
        
        import heapq
        
        def world_to_map(wx, wy):
            mx = int((wx - self.map_info.origin.position.x) / self.map_info.resolution)
            my = int((wy - self.map_info.origin.position.y) / self.map_info.resolution)
            return mx, my
        
        def map_to_world(mx, my):
            wx = mx * self.map_info.resolution + self.map_info.origin.position.x
            wy = my * self.map_info.resolution + self.map_info.origin.position.y
            return wx, wy
        
        def is_valid(mx, my):
            if mx < 0 or mx >= self.map_info.width or my < 0 or my >= self.map_info.height:
                return False
            return not self._inflated_map[my, mx]
        
        def heuristic(a, b):
            return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        
        start_map = world_to_map(start[0], start[1])
        goal_map = world_to_map(goal[0], goal[1])
        
        if (start_map[0] < 0 or start_map[0] >= self.map_info.width or
            start_map[1] < 0 or start_map[1] >= self.map_info.height):
            return []
        
        if (goal_map[0] < 0 or goal_map[0] >= self.map_info.width or
            goal_map[1] < 0 or goal_map[1] >= self.map_info.height):
            return []
        
        if not is_valid(goal_map[0], goal_map[1]):
            self.get_logger().warn('Goal is inside obstacle, searching nearest free cell')
            best_goal = None
            best_dist = float('inf')
            for r in range(1, 20):
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        nx, ny = goal_map[0] + dx, goal_map[1] + dy
                        if is_valid(nx, ny):
                            d = heuristic((nx, ny), goal_map)
                            if d < best_dist:
                                best_dist = d
                                best_goal = (nx, ny)
                if best_goal is not None:
                    break
            if best_goal is None:
                return []
            goal_map = best_goal
        
        SQRT2 = math.sqrt(2)
        neighbors = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),           # Cardinal
            (1, 1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (-1, -1, SQRT2)  # Diagonal
        ]
        
        open_set = []               # (f_score, counter, (mx, my))
        counter = 0
        heapq.heappush(open_set, (0.0, counter, start_map))
        
        came_from = {}
        g_score = {start_map: 0.0}
        f_score = {start_map: heuristic(start_map, goal_map)}
        closed_set = set()
        
        max_iterations = 50000      # Prevent infinite loops
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            _, _, current = heapq.heappop(open_set)
            
            if current == goal_map:
                path_map = []
                node = current
                while node in came_from:
                    path_map.append(node)
                    node = came_from[node]
                path_map.append(start_map)
                path_map.reverse()
                path_world = []
                for mx, my in path_map:
                    wx, wy = map_to_world(mx, my)
                    path_world.append([wx, wy])
                
                if len(path_world) > 2:
                    simplified = [path_world[0]]
                    for i in range(1, len(path_world) - 1):
                        prev = simplified[-1]
                        curr = path_world[i]
                        nxt = path_world[i + 1]
                        cross = (curr[0] - prev[0]) * (nxt[1] - prev[1]) - \
                                (curr[1] - prev[1]) * (nxt[0] - prev[0])
                        if abs(cross) > 1e-6:
                            simplified.append(curr)
                    simplified.append(path_world[-1])
                    return simplified
                
                return path_world
            
            if current in closed_set:
                continue
            closed_set.add(current)
            
            for dx, dy, move_cost in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if neighbor in closed_set:
                    continue
                
                if not is_valid(neighbor[0], neighbor[1]):
                    continue
                
                tentative_g = g_score[current] + move_cost
                
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal_map)
                    f_score[neighbor] = f
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))
        
        self.get_logger().warn(f'A* failed after {iterations} iterations, no path found')
        return []
    
    def distance(self, p1: Point, p2: Point) -> float:
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        dz = p2.z - p1.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def _precompute_inflated_map(self) -> None:
        if self.occupancy_map is None or self.map_info is None:
            return

        occupied = self.occupancy_map > 50  # bool mask
        inflate_cells = max(1, int(self.obstacle_inflation / self.map_info.resolution))
        size = 2 * inflate_cells + 1
        y_grid, x_grid = np.ogrid[-inflate_cells:inflate_cells + 1,
                                  -inflate_cells:inflate_cells + 1]
        kernel = (x_grid * x_grid + y_grid * y_grid) <= (inflate_cells * inflate_cells)

        try:
            from scipy.ndimage import binary_dilation
            self._inflated_map = binary_dilation(occupied, structure=kernel)
        except ImportError:
            h, w = occupied.shape
            inflated = np.zeros_like(occupied)
            ys, xs = np.where(occupied)
            for oy, ox in zip(ys, xs):
                y_lo = max(0, oy - inflate_cells)
                y_hi = min(h, oy + inflate_cells + 1)
                x_lo = max(0, ox - inflate_cells)
                x_hi = min(w, ox + inflate_cells + 1)
                inflated[y_lo:y_hi, x_lo:x_hi] = True
            self._inflated_map = inflated


def main(args=None):
    rclpy.init(args=args)
    node = PathPlanner()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

