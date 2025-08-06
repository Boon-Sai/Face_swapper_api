from src.exceptions import CustomException
from src.logger import logging
from src.constants import *

class ConfigEntity:
    def __init__(self):
        self.model_name = MODEL_NAME
        self.swapper_model_dir = SWAPPER_MODEL_DIR
        self.ctx_id = CTX_ID
        self.det_size = DET_SIZE
        self.output_dir = OUTPUT_DIR
        self.result_image_dir = RESULT_IMAGE_DIR
        self.detected_faces_dir = DETECTED_FACES_DIR

class ModelInitializerConfig:
    def __init__(self, config: ConfigEntity):
        self.model_name = config.model_name
        self.ctx_id = config.ctx_id
        self.det_size = config.det_size

class SwapperModelConfig:
    def __init__(self, config: ConfigEntity):
        self.swapper_model_dir = config.swapper_model_dir
        self.output_dir = config.output_dir
        self.result_image_dir = config.result_image_dir
        self.detected_faces_dir = config.detected_faces_dir