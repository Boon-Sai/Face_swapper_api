import cv2
import sys
from src.components.model_initializer import ModelInitializer
from src.components.faceswap import FaceSwap
from src.entity.face_swap_artifact import SwapperModelArtifact
from src.logger import logging
from src.exceptions import CustomException

def initiate_face_swapper(multi_face_img_path: str, single_face_img_path: str, selected_indices: list = None):
    try:
        logging.info("=== Starting Face Swap Pipeline ===")

        logging.info("Initializing FaceAnalysis model...")
        model_initializer = ModelInitializer()
        face_analysis_app = model_initializer.initialize_model()

        logging.info(f"Loading input images: {multi_face_img_path}, {single_face_img_path}")
        img_multi_faces = cv2.imread(multi_face_img_path)
        img_single_face = cv2.imread(single_face_img_path)

        if img_multi_faces is None:
            raise FileNotFoundError(f"Multi-face image not found: {multi_face_img_path}")
        if img_single_face is None:
            raise FileNotFoundError(f"Single-face image not found: {single_face_img_path}")

        logging.info("Input images loaded successfully.")

        face_swapper = FaceSwap()
        faces, face_paths, base64_faces = face_swapper.detect_and_save_faces(face_analysis_app, img_multi_faces)
        logging.info(f"Detected face paths: {face_paths}")

        artifact = SwapperModelArtifact(result_image_path="", detected_face_paths=face_paths, base64_faces=base64_faces)
        if selected_indices is None:
            logging.info("No indices provided; selecting all detected faces for swapping.")
            selected_indices = list(range(len(faces)))

        if not faces:
            logging.info("No faces to swap; returning detection artifact.")
            return artifact

        logging.info(f"Selected face indices: {selected_indices}")

        artifact = face_swapper.perform_face_swapping(
            app=face_analysis_app,
            img_multi_faces=img_multi_faces,
            img_single_face=img_single_face,
            selected_indices=selected_indices
        )
        artifact.detected_face_paths = face_paths
        artifact.base64_faces = base64_faces

        logging.info(f"Face swap completed. Result saved at: {artifact.result_image_path}")
        logging.info(f"Detected faces saved at: {artifact.detected_face_paths}")

        logging.info("=== Pipeline Completed Successfully ===")
        return artifact

    except Exception as e:
        logging.error("Pipeline execution failed.", exc_info=True)
        raise CustomException(e, sys) from e