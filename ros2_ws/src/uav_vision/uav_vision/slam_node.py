import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, PoseWithCovariance
from std_msgs.msg import Bool, Float64, Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque
import time
import math
import logging
from uav_vision.visual_odometry import VisualOdometry
from uav_vision.graph_optimizer import GraphOptimizer

logger = logging.getLogger(__name__)

class SLAMNode(Node):
    def __init__(self):
        super().__init__('slam_node')
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('odom_topic', '/uav/vision/odometry')
        self.declare_parameter('map_topic', '/uav/slam/map')
        self.declare_parameter('pose_topic', '/uav/slam/pose')
        self.declare_parameter('loop_closure_topic', '/uav/slam/loop_closure')
        self.declare_parameter('map_resolution', 0.1)   # meters per pixel
        self.declare_parameter('map_width', 200)        # pixels
        self.declare_parameter('map_height', 200)       # pixels
        self.declare_parameter('loop_closure_threshold', 0.5)       # meters
        self.declare_parameter('loop_closure_min_distance', 5.0)    # meters
        self.declare_parameter('loop_closure_history_size', 100)
        self.declare_parameter('bow_vocabulary_size', 1000)         # Number of visual words
        self.declare_parameter('bow_similarity_threshold', 0.3)     # Cosine similarity threshold
        self.declare_parameter('bow_min_feature_matches', 15)       # Min matches for verification
        self.declare_parameter('bow_max_distance', 10.0)            # Max distance for loop closure candidate (meters)
        
        camera_topic = self.get_parameter('camera_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        map_topic = self.get_parameter('map_topic').value
        pose_topic = self.get_parameter('pose_topic').value
        loop_closure_topic = self.get_parameter('loop_closure_topic').value
        
        self.visual_odometry = VisualOdometry(
            feature_detector="ORB",
            max_features=500,
            min_match_count=10
        )
        
        self.graph_optimizer = GraphOptimizer()
        self.use_graph_optimization = True
        self.optimization_interval = 10  # Optimize every N loop closures
        self.map_resolution = self.get_parameter('map_resolution').value
        self.map_width = self.get_parameter('map_width').value
        self.map_height = self.get_parameter('map_height').value
        self.occupancy_map = np.ones((self.map_height, self.map_width), dtype=np.int8) * -1  # Unknown
        self.height_map = np.full((self.map_height, self.map_width), float('nan'), dtype=np.float32)  # 2.5D height layer
        self.map_origin_x = self.map_width // 2
        self.map_origin_y = self.map_height // 2
        self.pose_history = deque(maxlen=self.get_parameter('loop_closure_history_size').value)
        self.keyframe_history = deque(maxlen=self.get_parameter('loop_closure_history_size').value)
        self.current_pose = np.array([0.0, 0.0, 0.0])       # x, y, yaw
        self.current_z = 0.0                                # Track altitude separately (meters)
        self.current_rotation = np.eye(3)
        self.loop_closure_threshold = self.get_parameter('loop_closure_threshold').value
        self.loop_closure_min_distance = self.get_parameter('loop_closure_min_distance').value
        self.last_loop_closure_pose = None
        self.bow_vocabulary_size = self.get_parameter('bow_vocabulary_size').value
        self.bow_similarity_threshold = self.get_parameter('bow_similarity_threshold').value
        self.bow_min_feature_matches = self.get_parameter('bow_min_feature_matches').value
        self.bow_max_distance = self.get_parameter('bow_max_distance').value
        self.bow_vocabulary = None              # Will be built incrementally
        self.bow_descriptors_history = deque(maxlen=self.get_parameter('loop_closure_history_size').value)
        self.keyframe_features_history = deque(maxlen=self.get_parameter('loop_closure_history_size').value)
        self._keyframe_heights = deque(maxlen=self.get_parameter('loop_closure_history_size').value)
        self.bow_descriptor_accumulator = []    # Raw descriptors for vocabulary building
        self.bow_vocabulary_built = False
        self.bow_build_threshold = 30           # Build vocabulary after N keyframes
        self.loop_closure_count = 0
        self._keyframe_counter = 0              # Monotonic counter for graph pose IDs
        self.bow_detector = cv2.ORB_create(nfeatures=500)
        self.bow_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.bridge = CvBridge()
        
        self.image_sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10
        )
        
        self.map_pub = self.create_publisher(OccupancyGrid, map_topic, 10)
        self.height_map_pub = self.create_publisher(Float32MultiArray, map_topic + '/height', 10)
        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)
        self.odom_pub = self.create_publisher(Odometry, odom_topic, 10)
        self.loop_closure_pub = self.create_publisher(Bool, loop_closure_topic, 10)
        self.map_timer = self.create_timer(1.0, self.publish_map)
        self.get_logger().info('SLAM Node started (2.5D height-aware)')
        self.get_logger().info(f'Map resolution: {self.map_resolution}m/pixel')
        self.get_logger().info(f'Map size: {self.map_width}x{self.map_height}')
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            success, translation, rotation, covariance = self.visual_odometry.process_frame(cv_image)
            
            if success:
                self.update_pose(translation, rotation)
                loop_detected = self.detect_loop_closure()
                
                if loop_detected:
                    self.get_logger().info('Loop closure detected!')
                    self.correct_pose_with_loop_closure()
                    if self.use_graph_optimization:
                        self.add_loop_closure_to_graph()

                self.publish_pose(msg.header.stamp)
                self.publish_odometry(msg.header.stamp, translation, rotation, covariance)
                self.store_keyframe(cv_image, self.current_pose.copy())
                self.update_map()
        
        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')
    
    def update_pose(self, translation: np.ndarray, rotation: np.ndarray):
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        translation_world = self.current_rotation @ translation
        self.current_pose[0] += translation_world[0]
        self.current_pose[1] += translation_world[1]
        self.current_pose[2] = yaw
        if len(translation) > 2:
            self.current_z += float(translation_world[2])
        self.current_rotation = self.current_rotation @ rotation
    
    def detect_loop_closure(self) -> bool:
        if len(self.pose_history) < 10:
            return False
        
        current_pos = self.current_pose[:2]         # x, y only
        current_z = self.current_z                  # Height for 3D distance check
        
        if self.last_loop_closure_pose is not None:
            dist_from_last = np.linalg.norm(current_pos - self.last_loop_closure_pose)
            if dist_from_last < self.loop_closure_min_distance:
                return False                        # Too close to last loop closure

        if self.bow_vocabulary_built and len(self.bow_descriptors_history) > 20:
            current_bow = self.bow_descriptors_history[-1] if self.bow_descriptors_history else None
            if current_bow is not None:
                best_similarity = 0.0
                best_idx = -1
                
                for i in range(len(self.bow_descriptors_history) - 20):  # Skip recent
                    past_bow = self.bow_descriptors_history[i]
                    if past_bow is None:
                        continue

                    if i < len(self.pose_history):
                        past_pos = self.pose_history[i][:2]
                        distance = np.linalg.norm(current_pos - past_pos)
                        if distance > self.bow_max_distance:
                            continue

                    similarity = self._bow_similarity(current_bow, past_bow)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_idx = i
                
                if best_similarity > self.bow_similarity_threshold and best_idx >= 0:
                    if self._verify_loop_closure(best_idx):
                        self.get_logger().info(
                            f'Visual loop closure detected! similarity={best_similarity:.3f}, '
                            f'keyframe_idx={best_idx}, distance={np.linalg.norm(current_pos - self.pose_history[best_idx][:2]):.2f}m'
                        )
                        self.loop_closure_count += 1
                        return True
        
        for i, past_pose in enumerate(self.pose_history):
            if i < 20:  
                continue
            
            past_pos = past_pose[:2]
            distance = np.linalg.norm(current_pos - past_pos)
            
            if i < len(self._keyframe_heights):
                dz = abs(current_z - self._keyframe_heights[i])
                distance_3d = math.sqrt(distance**2 + dz**2)
            else:
                distance_3d = distance
            
            if distance_3d < self.loop_closure_threshold:
                if len(self.keyframe_features_history) > 0 and i < len(self.keyframe_features_history):
                    if self._verify_loop_closure(i):
                        self.get_logger().info(
                            f'Distance+visual loop closure: distance={distance:.2f}m'
                        )
                        self.loop_closure_count += 1
                        return True
                else:
                    self.get_logger().info(
                        f'Distance-only loop closure: distance={distance:.2f}m'
                    )
                    self.loop_closure_count += 1
                    return True
        
        return False
    
    def _bow_similarity(self, bow1: np.ndarray, bow2: np.ndarray) -> float:
        norm1 = np.linalg.norm(bow1)
        norm2 = np.linalg.norm(bow2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        return float(np.dot(bow1, bow2) / (norm1 * norm2))
    
    def _verify_loop_closure(self, candidate_idx: int) -> bool:
        if not self.keyframe_features_history:
            return True  
        if candidate_idx >= len(self.keyframe_features_history):
            return True
        
        current_features = self.keyframe_features_history[-1] if self.keyframe_features_history else None
        candidate_features = self.keyframe_features_history[candidate_idx]
        
        if current_features is None or candidate_features is None:
            return True
        
        current_kp, current_desc = current_features
        candidate_kp, candidate_desc = candidate_features
        
        if current_desc is None or candidate_desc is None:
            return True
        
        if len(current_desc) < 2 or len(candidate_desc) < 2:
            return True
        
        try:
            knn_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = knn_matcher.knnMatch(current_desc, candidate_desc, k=2)
            
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance: 
                        good_matches.append(m)
            
            verified = len(good_matches) >= self.bow_min_feature_matches
            if verified:
                self.get_logger().debug(
                    f'Loop closure verified: {len(good_matches)} good matches'
                )
            return verified
        except Exception as e:
            logger.debug(f'Feature verification failed: {e}')
            return False
    
    def _compute_bow_descriptor(self, descriptors: np.ndarray) -> np.ndarray:
        if self.bow_vocabulary is None or descriptors is None or len(descriptors) == 0:
            return None
        
        bow_vector = np.zeros(self.bow_vocabulary_size, dtype=np.float32)
        
        for desc in descriptors:
            distances = np.linalg.norm(self.bow_vocabulary.astype(np.float32) - desc.astype(np.float32), axis=1)
            nearest_word = np.argmin(distances)
            bow_vector[nearest_word] += 1.0
        norm = np.linalg.norm(bow_vector)
        if norm > 0:
            bow_vector /= norm
        
        return bow_vector
    
    def _build_bow_vocabulary(self):
        if len(self.bow_descriptor_accumulator) < self.bow_build_threshold:
            return
        
        self.get_logger().info(
            f'Building BoW vocabulary from {len(self.bow_descriptor_accumulator)} keyframes...'
        )
        
        all_descriptors = []
        for desc in self.bow_descriptor_accumulator:
            if desc is not None and len(desc) > 0:
                all_descriptors.append(desc)
        
        if not all_descriptors:
            return
        
        all_desc_array = np.vstack(all_descriptors).astype(np.float32)
        actual_k = min(self.bow_vocabulary_size, len(all_desc_array) // 2)
        if actual_k < 10:
            return  
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
        try:
            _, labels, centers = cv2.kmeans(
                all_desc_array, actual_k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
            )
            self.bow_vocabulary = centers
            self.bow_vocabulary_size = actual_k  # Update to actual size
            self.bow_vocabulary_built = True
            self.bow_descriptors_history = deque(maxlen=self.pose_history.maxlen)
            for desc in self.bow_descriptor_accumulator:
                bow = self._compute_bow_descriptor(desc)
                self.bow_descriptors_history.append(bow)
            
            self.get_logger().info(
                f'BoW vocabulary built: {actual_k} visual words from {len(all_desc_array)} descriptors'
            )
        except Exception as e:
            self.get_logger().error(f'Failed to build BoW vocabulary: {e}')
    
    def correct_pose_with_loop_closure(self):
        current_pos = self.current_pose[:2]
        min_dist = float('inf')
        closest_pose = None
        
        for past_pose in self.pose_history:
            past_pos = past_pose[:2]
            distance = np.linalg.norm(current_pos - past_pos)
            
            if distance < min_dist and distance < self.loop_closure_threshold:
                min_dist = distance
                closest_pose = past_pose
        
        if closest_pose is not None:
            alpha = 0.3  # Correction weight
            self.current_pose[0] = alpha * closest_pose[0] + (1 - alpha) * self.current_pose[0]
            self.current_pose[1] = alpha * closest_pose[1] + (1 - alpha) * self.current_pose[1]
            self.last_loop_closure_pose = self.current_pose[:2].copy()
            msg = Bool()
            msg.data = True
            self.loop_closure_pub.publish(msg)
    
    def add_loop_closure_to_graph(self):
        if len(self.pose_history) < 2:
            return
        
        current_idx = self._keyframe_counter - 1
        current_pose = self.current_pose
        min_dist = float('inf')
        closest_idx = None
        
        pose_list = list(self.pose_history)
        for i, past_pose in enumerate(pose_list[:-1]):
            if i < 20:  # Skip recent
                continue
            distance = np.linalg.norm(current_pose[:2] - past_pose[:2])
            if distance < min_dist and distance < self.loop_closure_threshold:
                min_dist = distance
                closest_idx = i
        
        if closest_idx is not None:
            base_id = self._keyframe_counter - len(self.pose_history)
            pose_id_i = f"pose_{base_id + closest_idx}"
            pose_id_j = f"pose_{current_idx}"
            
            if pose_id_i not in self.graph_optimizer.node_indices:
                self.graph_optimizer.add_pose(pose_id_i, self.pose_history[closest_idx])
            if pose_id_j not in self.graph_optimizer.node_indices:
                self.graph_optimizer.add_pose(pose_id_j, current_pose)

            relative_pose = current_pose - self.pose_history[closest_idx]
            self.graph_optimizer.add_loop_closure(pose_id_i, pose_id_j, relative_pose)
            
            if len(self.graph_optimizer.pose_graph.loop_closures) % self.optimization_interval == 0:
                optimized_poses = self.graph_optimizer.optimize(iterations=5)
                for pose_id, optimized_pose in optimized_poses.items():
                    idx = int(pose_id.split('_')[1]) - base_id
                    if 0 <= idx < len(self.pose_history):
                        self.pose_history[idx] = optimized_pose.copy()
    
    def store_keyframe(self, image: np.ndarray, pose: np.ndarray):
        self.pose_history.append(pose.copy())
        self._keyframe_heights.append(float(self.current_z))
        self._keyframe_counter += 1

        if self.use_graph_optimization:
            pose_id = f"pose_{self._keyframe_counter - 1}"
            self.graph_optimizer.add_pose(pose_id, pose)
            
            if self._keyframe_counter > 1:
                prev_pose_id = f"pose_{self._keyframe_counter - 2}"
                relative_pose = pose - self.pose_history[-2]
                self.graph_optimizer.add_odometry(prev_pose_id, pose_id, relative_pose)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        resized = cv2.resize(gray, (64, 48))  # Small thumbnail
        self.keyframe_history.append(resized)
 
        try:
            keypoints, descriptors = self.bow_detector.detectAndCompute(gray, None)
            
            if keypoints and descriptors is not None:
                kp_data = [(kp.pt, kp.size, kp.angle) for kp in keypoints]
                self.keyframe_features_history.append((kp_data, descriptors))
            else:
                self.keyframe_features_history.append(None)

            if descriptors is not None:
                self.bow_descriptor_accumulator.append(descriptors)

            if not self.bow_vocabulary_built and len(self.bow_descriptor_accumulator) >= self.bow_build_threshold:
                self._build_bow_vocabulary()
 
            if self.bow_vocabulary_built and descriptors is not None:
                bow_desc = self._compute_bow_descriptor(descriptors)
                self.bow_descriptors_history.append(bow_desc)
            elif self.bow_vocabulary_built:
                self.bow_descriptors_history.append(None)
        except Exception as e:
            self.get_logger().debug(f'BoW feature extraction failed: {e}')
            self.keyframe_features_history.append(None)
            if self.bow_vocabulary_built:
                self.bow_descriptors_history.append(None)
    
    def update_map(self):
        map_x = int(self.map_origin_x + self.current_pose[0] / self.map_resolution)
        map_y = int(self.map_origin_y + self.current_pose[1] / self.map_resolution)

        if 0 <= map_x < self.map_width and 0 <= map_y < self.map_height:
            radius = 3  # pixels
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    px = map_x + dx
                    py = map_y + dy
                    if 0 <= px < self.map_width and 0 <= py < self.map_height:
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist <= radius:
                            self.occupancy_map[py, px] = 0
                            current_height = float(self.current_z)
                            if math.isnan(self.height_map[py, px]):
                                self.height_map[py, px] = current_height
                            else:
                                self.height_map[py, px] = max(
                                    self.height_map[py, px], current_height
                                )
    
    def publish_map(self):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.info.resolution = self.map_resolution
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        msg.info.origin.position.x = -self.map_origin_x * self.map_resolution
        msg.info.origin.position.y = -self.map_origin_y * self.map_resolution
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = self.occupancy_map.flatten().tolist()
        
        self.map_pub.publish(msg)
        
        height_msg = Float32MultiArray()
        height_data = self.height_map.copy()
        height_data[np.isnan(height_data)] = -1.0
        height_msg.data = height_data.flatten().tolist()
        self.height_map_pub.publish(height_msg)
    
    def publish_pose(self, stamp):
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        msg.pose.position.x = float(self.current_pose[0])
        msg.pose.position.y = float(self.current_pose[1])
        msg.pose.position.z = float(self.current_z)

        yaw = self.current_pose[2]
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        
        self.pose_pub.publish(msg)
    
    def publish_odometry(self, stamp, translation, rotation, covariance):
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = float(self.current_pose[0])
        msg.pose.pose.position.y = float(self.current_pose[1])
        msg.pose.pose.position.z = float(self.current_z)

        yaw = self.current_pose[2]
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        if covariance is not None:
            cov_flat = covariance.flatten()
            if len(cov_flat) >= 36:
                msg.pose.covariance = [float(x) for x in cov_flat[:36]]
        
        self.odom_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SLAMNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

