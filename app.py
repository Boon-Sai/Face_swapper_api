from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from src.pipeline.faceswap_pipeline import initiate_face_swapper
from src.logger import logging
from src.exceptions import CustomException
import os
import sys
from typing import List

app = FastAPI()

@app.post("/upload-images/")
async def upload_images(multi_face_image: UploadFile = File(...), single_face_image: UploadFile = File(...)):
    try:
        multi_face_path = os.path.join("artifacts", multi_face_image.filename)
        single_face_path = os.path.join("artifacts", single_face_image.filename)
        os.makedirs("artifacts", exist_ok=True)

        with open(multi_face_path, "wb") as f:
            f.write(await multi_face_image.read())
        with open(single_face_path, "wb") as f:
            f.write(await single_face_image.read())

        logging.info(f"Calling initiate_face_swapper with {multi_face_path}, {single_face_path}")
        face_swapper = initiate_face_swapper(multi_face_path, single_face_path, selected_indices=None)
        
        if not face_swapper.detected_face_paths:
            return {
                "message": "No faces detected in the multi-face image. Please try different images.",
                "detected_faces": [],
                "multi_face_path": multi_face_path,
                "single_face_path": single_face_path
            }

        detected_faces = [
            {"index": i + 1, "base64": f"data:image/jpeg;base64,{base64_face}", "path": path}
            for i, (base64_face, path) in enumerate(zip(face_swapper.base64_faces, face_swapper.detected_face_paths))
        ]
        return {
            "message": f"Detected {len(face_swapper.detected_face_paths)} faces. Please select faces to swap by index (e.g., '1,3' or '-1' for all) in the /swap-faces/ endpoint.",
            "detected_faces": detected_faces,
            "multi_face_path": multi_face_path,
            "single_face_path": single_face_path
        }
    except Exception as e:
        logging.error(f"Error in image upload endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/swap-faces/")
async def swap_faces(multi_face_path: str, single_face_path: str, indices: str):
    try:
        if indices == "-1":
            selected_indices = None  # Swap all faces
        else:
            try:
                selected_indices = [int(i) - 1 for i in indices.split(",") if i.strip().isdigit()]
                if not selected_indices:
                    raise ValueError("Invalid indices provided")
            except ValueError as e:
                logging.warning(f"Invalid indices input: {indices}. Defaulting to all faces.")
                selected_indices = None

        artifact = initiate_face_swapper(multi_face_path, single_face_path, selected_indices)
        return {
            "result_image_path": artifact.result_image_path,
            "detected_face_paths": artifact.detected_face_paths
        }
    except Exception as e:
        logging.error(f"Error in face swap endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/download-file/{file_path:path}")
async def download_file(file_path: str):
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")