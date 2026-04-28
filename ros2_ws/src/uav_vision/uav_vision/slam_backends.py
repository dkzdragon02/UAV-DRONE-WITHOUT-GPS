class ORBSLAM3Wrapper:
    def __init__(self, vocab_path: str, settings_path: str):
        self.vocab_path = vocab_path
        self.settings_path = settings_path

    def process(self, image):
        return False, None, None

class OpenVINSWrapper:
    def __init__(self, config_path: str):
        self.config_path = config_path

    def process(self, image):
        return False, None, None

