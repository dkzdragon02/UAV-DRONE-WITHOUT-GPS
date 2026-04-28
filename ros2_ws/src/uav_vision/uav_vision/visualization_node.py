import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import ColorRGBA
import numpy as np

class VisualizationNode(Node): 
    def __init__(self):
        super().__init__('visualization_node')

        self.declare_parameter('path_topic', '/uav/path_planner/path')
        self.declare_parameter('waypoint_topic', '/uav/path_planner/waypoints')
        self.declare_parameter('markers_topic', '/uav/visualization/markers')
        
        path_topic = self.get_parameter('path_topic').value
        waypoint_topic = self.get_parameter('waypoint_topic').value
        markers_topic = self.get_parameter('markers_topic').value

        self.current_path = None
        self.waypoints = []
        
        self.path_sub = self.create_subscription(
            Path,
            path_topic,
            self.path_callback,
            10
        )
        
        self.waypoint_sub = self.create_subscription(
            Path,
            waypoint_topic,
            self.waypoint_callback,
            10
        )
        
        self.markers_pub = self.create_publisher(MarkerArray, markers_topic, 10)
        self.viz_timer = self.create_timer(0.5, self.publish_markers)
        self.get_logger().info('Visualization Node started')
    
    def path_callback(self, msg):
        self.current_path = msg
    
    def waypoint_callback(self, msg):
        self.waypoints = msg.poses
    
    def publish_markers(self):
        marker_array = MarkerArray()
        for i, waypoint in enumerate(self.waypoints):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose = waypoint.pose
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker_array.markers.append(marker)
            
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.id = i + 1000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose = waypoint.pose
            text_marker.pose.position.z += 0.5
            text_marker.scale.z = 0.3
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.text = f"WP{i+1}"
            marker_array.markers.append(text_marker)
        
        if self.current_path and len(self.current_path.poses) > 0:
            path_marker = Marker()
            path_marker.header.frame_id = "map"
            path_marker.header.stamp = self.get_clock().now().to_msg()
            path_marker.id = 2000
            path_marker.type = Marker.LINE_STRIP
            path_marker.action = Marker.ADD
            path_marker.scale.x = 0.1
            path_marker.color.r = 0.0
            path_marker.color.g = 1.0
            path_marker.color.b = 0.0
            path_marker.color.a = 1.0
            
            for pose_stamped in self.current_path.poses:
                point = Point()
                point.x = pose_stamped.pose.position.x
                point.y = pose_stamped.pose.position.y
                point.z = pose_stamped.pose.position.z
                path_marker.points.append(point)
            
            marker_array.markers.append(path_marker)
        
        if marker_array.markers:
            self.markers_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

