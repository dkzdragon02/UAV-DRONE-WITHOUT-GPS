import cv2
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class CameraHandler:
    def __init__(self, 
                width: int = 640,
                height: int = 480,
                fps: int = 30,
                device: int = 0,
                use_picamera: bool = True,
                calibration_file: Optional[str] = None):
        self.width = width
        self.height = height
        self.fps = fps
        self.device = device
        self.use_picamera = use_picamera
        self.calibration_file = calibration_file
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera = None
        self.is_initialized = False
        self._initialize_camera()
        self._load_calibration()
    
    def _initialize_camera(self):
        try:
            if self.use_picamera:
                try:
                    from picamera2 import Picamera2
                    self.camera = Picamera2()
                    config = self.camera.create_preview_configuration(
                        main={"size": (self.width, self.height), "format": "RGB888"}
                    )
                    self.camera.configure(config)
                    self.camera.start()
                    
                    logger.info(f"Picamera2 initialized: {self.width}x{self.height}@{self.fps}fps")
                    self.is_initialized = True
                    
                except ImportError:
                    logger.warning("picamera2 not available, falling back to USB camera")
                    self.use_picamera = False
                    self._initialize_usb_camera()
                except Exception as e:
                    logger.error(f"Failed to initialize picamera2: {e}")
                    self.use_picamera = False
                    self._initialize_usb_camera()
            else:
                self._initialize_usb_camera()
                
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            self.is_initialized = False
    
    def _initialize_usb_camera(self):
        try:
            self.camera = cv2.VideoCapture(self.device)
            
            if not self.camera.isOpened():
                raise Exception(f"Could not open camera device {self.device}")
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            
            logger.info(f"USB camera initialized: {self.width}x{self.height}@{self.fps}fps")
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize USB camera: {e}")
            self.is_initialized = False
    
    def read_frame(self) -> Optional[np.ndarray]:
        if not self.is_initialized:
            return None
        try:
            if self.use_picamera:
                frame = self.camera.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = self.camera.read()
                if not ret:
                    return None
            
            return frame
            
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return None
    
    def release(self):
        if self.camera is not None:
            if self.use_picamera:
                self.camera.stop()
                self.camera.close()
            else:
                self.camera.release()
            self.is_initialized = False
            logger.info("Camera released")
    
    def get_camera_matrix(self) -> Optional[np.ndarray]:
        if self.camera_matrix is not None:
            return self.camera_matrix
        fx = self.width * 0.7
        fy = self.height * 0.7
        cx = self.width / 2.0
        cy = self.height / 2.0
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        return self.camera_matrix
    
    def get_distortion_coeffs(self) -> Optional[np.ndarray]:
        if self.dist_coeffs is not None:
            return self.dist_coeffs
        self.dist_coeffs = np.zeros(5, dtype=np.float32)
        return self.dist_coeffs

    def _load_calibration(self):
        if not self.calibration_file:
            return
        try:
            import yaml
            with open(self.calibration_file, "r") as f:
                data = yaml.safe_load(f)
            k = data.get("camera_matrix") or data.get("K")
            d = data.get("dist_coeffs") or data.get("D")
            if k is not None:
                self.camera_matrix = np.array(k, dtype=np.float32).reshape(3, 3)
            if d is not None:
                self.dist_coeffs = np.array(d, dtype=np.float32).flatten()
            logger.info(f"Loaded camera calibration from {self.calibration_file}")
        except Exception as e:
            logger.warning(f"Failed to load calibration {self.calibration_file}: {e}")

