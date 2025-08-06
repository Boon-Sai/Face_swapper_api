from src.entity.face_swap_config import ConfigEntity, ModelInitializerConfig
from src.entity.face_swap_artifact import ModelInitializationArtifact
from src.exceptions import CustomException
from src.logger import logging

import sys
from insightface.app import FaceAnalysis

class ModelInitializer:
    """
    Initializes the FaceAnalysis model using configuration.
    """
    def __init__(self):
        try:
            logging.info("Creating ModelInitializerConfig...")
            self.model_initializer_config = ModelInitializerConfig(config=ConfigEntity())
            logging.info(f"ModelInitializerConfig initialized with model name: {self.model_initializer_config.model_name}")
        except Exception as e:
            logging.error("Failed to initialize ModelInitializerConfig", exc_info=True)
            raise CustomException(e, sys) from e

    def initialize_model(self) -> FaceAnalysis:
        """
        Initializes and prepares the FaceAnalysis model.
        Returns:
            FaceAnalysis: The prepared FaceAnalysis model.
        """
        try:
            logging.info(f"Starting FaceAnalysis model initialization with model: {self.model_initializer_config.model_name}...")
            app = FaceAnalysis(name=self.model_initializer_config.model_name)
            app.prepare(
                ctx_id=self.model_initializer_config.ctx_id,
                det_size=self.model_initializer_config.det_size
            )
            logging.info("FaceAnalysis model initialized successfully.")

            # Optionally track artifact (not returned)
            artifact = ModelInitializationArtifact(
                model_name=self.model_initializer_config.model_name
            )
            logging.info(f"ModelInitializationArtifact created: {artifact}")

            return app

        except Exception as e:
            logging.error("Error during model initialization", exc_info=True)
            raise CustomException(e, sys) from e