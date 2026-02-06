"""
Stub wrappers for ORB-SLAM3 and OpenVINS.
Fill in initialization with actual binaries/libraries when available.
"""

class ORBSLAM3Wrapper:
    def __init__(self, vocab_path: str, settings_path: str):
        # TODO: integrate ORB-SLAM3 Python binding or bridge
        self.vocab_path = vocab_path
        self.settings_path = settings_path

    def process(self, image):
        # Should return (success, position, orientation_quat)
        return False, None, None


class OpenVINSWrapper:
    def __init__(self, config_path: str):
        # TODO: integrate OpenVINS ROS2 or python bindings
        self.config_path = config_path

    def process(self, image):
        return False, None, None

