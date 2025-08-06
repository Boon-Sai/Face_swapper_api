from src.exceptions import CustomException
from src.logger import logging
from src.entity.face_swap_config import ConfigEntity, SwapperModelConfig
from src.entity.face_swap_artifact import SwapperModelArtifact
from src.utils import download_weights_from_google_drive

import insightface
import sys
import cv2
import os
import matplotlib.pyplot as plt
from datetime import datetime
import base64
import io


class FaceSwap:
    def __init__(self):
        try:
            logging.info("Creating SwapperModelConfig...")
            self.swapper_model_config = SwapperModelConfig(config=ConfigEntity())
            logging.info(f"SwapperModelConfig initialized with model path: {self.swapper_model_config.swapper_model_dir}")
        except Exception as e:
            logging.error("Failed to initialize SwapperModelConfig", exc_info=True)
            raise CustomException(e, sys) from e

    def detect_and_save_faces(self, app, img_multi_faces) -> tuple[list, list, list]:
        try:
            if img_multi_faces is None:
                logging.error("Multi-face image is None")
                raise ValueError("Multi-face image is None")
            logging.info("Converting multi-face image to RGB format...")
            img_multi_faces_rgb = cv2.cvtColor(img_multi_faces, cv2.COLOR_BGR2RGB)
            logging.info("Detecting faces in multi-face image...")
            faces = app.get(img_multi_faces_rgb)
            if not faces:
                logging.info("No faces detected in the multi-face image.")
                return [], [], []

            logging.info(f"{len(faces)} face(s) detected in multi-face image.")
            detected_faces_dir = os.path.join(self.swapper_model_config.output_dir, self.swapper_model_config.detected_faces_dir)
            os.makedirs(detected_faces_dir, exist_ok=True)

            face_paths = []
            base64_faces = []
            for i, face in enumerate(faces):
                bbox = face['bbox']
                bbox = [int(b) for b in bbox]
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    logging.warning(f"Invalid bounding box for face {i+1}: {bbox}. Skipping.")
                    continue
                face_img = img_multi_faces_rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                if face_img.size == 0:
                    logging.warning(f"Empty face image for face {i+1}. Skipping.")
                    continue
                face_path = os.path.join(detected_faces_dir, f"face_{i+1}.jpg")
                plt.imsave(face_path, face_img)
                logging.info(f"Saved face {i+1} to {face_path}")

                success, buffer = cv2.imencode('.jpg', cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                if not success:
                    logging.warning(f"Failed to encode face {i+1} to JPEG. Skipping base64 conversion.")
                    continue
                base64_face = base64.b64encode(buffer).decode('utf-8')
                base64_faces.append(base64_face)
                face_paths.append(face_path)

            return faces, face_paths, base64_faces
        except Exception as e:
            logging.error("Error during face detection and saving", exc_info=True)
            raise CustomException(e, sys) from e

    def perform_face_swapping(self, app, img_multi_faces, img_single_face, selected_indices: list) -> SwapperModelArtifact:
        try:
            model_path = self.swapper_model_config.swapper_model_dir
            if not os.path.exists(model_path):
                logging.info("Downloading face swapper model weights...")
                download_weights_from_google_drive()
            
            logging.info(f"Loading face swapper model from: {model_path}")
            swapper = insightface.model_zoo.get_model(model_path, download=False)
            logging.info("Face swapper model loaded successfully.")

            logging.info("Converting images to RGB format...")
            img_multi_faces_rgb = cv2.cvtColor(img_multi_faces, cv2.COLOR_BGR2RGB)
            img_single_face_rgb = cv2.cvtColor(img_single_face, cv2.COLOR_BGR2RGB)

            logging.info("Detecting faces in multi-face image...")
            faces_multi = app.get(img_multi_faces_rgb)
            logging.info("Detecting face in single-face image...")
            faces_single = app.get(img_single_face_rgb)

            if not faces_multi:
                logging.warning("No faces detected in the multi-face image.")
                raise ValueError("No faces detected in the multi-face image!")
            if not faces_single:
                logging.warning("No faces detected in the single-face image.")
                raise ValueError("No faces detected in the single-face image!")

            logging.info(f"{len(faces_multi)} face(s) detected in multi-face image.")
            logging.info(f"{len(faces_single)} face(s) detected in single-face image.")

            source_face = faces_single[0]
            result_image = img_multi_faces_rgb.copy()
            logging.info(f"Performing face swap for indices: {selected_indices}")
            for idx in selected_indices:
                if idx < len(faces_multi):
                    logging.info(f"Swapping face {idx + 1}...")
                    result_image = swapper.get(result_image, faces_multi[idx], source_face, paste_back=True)
                else:
                    logging.warning(f"Index {idx} is out of range. Skipping.")

            result_dir = os.path.join(self.swapper_model_config.output_dir, self.swapper_model_config.result_image_dir)
            os.makedirs(result_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_image_path = os.path.join(result_dir, f"swapped_face_{timestamp}.jpg")
            result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(result_image_path, result_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            logging.info(f"Swapped image saved to: {result_image_path}")

            return SwapperModelArtifact(result_image_path=result_image_path, detected_face_paths=[], base64_faces=[])
        except Exception as e:
            logging.error("Error during顔 swapping operation", exc_info=True)
            raise CustomException(e, sys) from e