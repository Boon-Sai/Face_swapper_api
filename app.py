from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from src.pipeline.faceswap_pipeline import initiate_face_swapper
from src.logger import logging
from src.exceptions import CustomException
import os
import sys
from typing import List

app = FastAPI()

# Global variable to store the latest upload data (for demonstration; use a proper session store in production)
latest_session_data = {}

@app.post("/upload-images/")
async def upload_images(multi_face_image: UploadFile = File(...), single_face_image: UploadFile = File(...)):
    try:
        # Declare global variable at the start of the function
        global latest_session_data
        
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
            # Update session data
            latest_session_data = {
                "multi_face_path": multi_face_path,
                "single_face_path": single_face_path,
                "detected_faces": []
            }
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
        
        # Update session data
        latest_session_data = {
            "multi_face_path": multi_face_path,
            "single_face_path": single_face_path,
            "detected_faces": detected_faces
        }
        
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
async def swap_faces(indices: str):
    try:
        # Retrieve latest session data
        if not latest_session_data:
            logging.error("No session data available. Please upload images first.")
            raise HTTPException(status_code=404, detail="No session data available. Please upload images first.")
        
        multi_face_path = latest_session_data["multi_face_path"]
        single_face_path = latest_session_data["single_face_path"]
        detected_faces = latest_session_data["detected_faces"]
        
        # Validate indices
        if indices == "-1":
            selected_indices = None  # Swap all faces
        else:
            try:
                selected_indices = [int(i) - 1 for i in indices.split(",") if i.strip().isdigit()]
                if not selected_indices:
                    raise ValueError("Invalid indices provided")
                # Validate indices against detected faces
                max_index = len(detected_faces)
                if any(i < 0 or i >= max_index for i in selected_indices):
                    raise ValueError(f"Indices out of range. Valid range: 1 to {max_index}")
            except ValueError as e:
                logging.warning(f"Invalid indices input: {indices}. Error: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid indices: {str(e)}")

        logging.info(f"Calling initiate_face_swapper with {multi_face_path}, {single_face_path}, indices: {indices}")
        artifact = initiate_face_swapper(multi_face_path, single_face_path, selected_indices)
        
        if not artifact.result_image_path or not os.path.exists(artifact.result_image_path):
            logging.error(f"Face swap failed. Result image not found at: {artifact.result_image_path}")
            raise HTTPException(status_code=500, detail="Face swap failed. Result image not generated.")

        # Return the face-swapped image as a direct download
        return FileResponse(
            path=artifact.result_image_path,
            filename=os.path.basename(artifact.result_image_path),
            media_type="image/jpeg"
        )
    except Exception as e:
        logging.error(f"Error in face swap endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/download-file/{file_path:path}")
async def download_file(file_path: str):
    if os.path.exists(file_path):
        return FileResponse(
            path=file_path,
            filename=os.path.basename(file_path),
            media_type="image/jpeg"
        )
    raise HTTPException(status_code=404, detail="File not found")