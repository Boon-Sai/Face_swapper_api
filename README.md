# Face Swapping with InsightFace & FastAPI

This project provides a REST API for face swapping between two images using the InsightFace library, leveraging state-of-the-art face analysis and swapping capabilities. It detects faces in a target image and replaces them with a face from a source image, supporting both CPU and GPU acceleration through a FastAPI web service.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Clone the Repository](#clone-the-repository)
  - [Set Up Virtual Environment](#set-up-virtual-environment)
  - [Install Dependencies](#install-dependencies)
  - [Download Weights File](#download-weights-file)
- [Usage](#usage)
  - [Start the API Server](#start-the-api-server)
  - [API Endpoints](#api-endpoints)
  - [Testing the API](#testing-the-api)
- [Code Structure](#code-structure)
- [Customization](#customization)
  - [Adjust Detection Thresholds](#adjust-detection-thresholds)
  - [Configure CORS Settings](#configure-cors-settings)
  - [Modify Upload Directory](#modify-upload-directory)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contact](#contact)

## Features
- RESTful API for face swapping operations
- Accurate face detection in images using InsightFace
- Face swapping from a source image onto multiple faces in a target image
- Support for GPU and CPU computation via configurable `ctx_id`
- CORS support for web applications
- Modular pipeline architecture for easy extension
- Comprehensive logging and exception handling
- File upload handling with unique naming

## Project Structure
```
FACE_SWAPPER_IMAGE/
├── src/
│   ├── __pycache__/
│   ├── components/
│   ├── constants/
│   ├── entity/
│   ├── exceptions/
│   ├── logger/
│   ├── pipeline/
│   ├── utils/
│   └── __init__.py
├── uploads/                    # Directory for uploaded images
├── data/                      # Data storage
├── logs/                      # Application logs
├── Artifacts/                 # Build artifacts
├── venv/                      # Virtual environment
├── weights/                   # Model weights directory
├── app.py                     # FastAPI application entry point
├── main.py                    # Alternative entry point
├── insightface.ipynb         # Jupyter notebook for testing
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
└── .gitignore               # Git ignore file
```

## Prerequisites
- Python 3.8 or higher
- Git
- Compatible operating system (Windows, Linux, or macOS)
- For GPU acceleration: NVIDIA GPU with CUDA support and appropriate drivers
- Images in supported formats (e.g., JPG, PNG)

## Installation

### Clone the Repository
Clone the project repository to your local machine:
```bash
git clone https://github.com/yourusername/face-swapping.git
cd face-swapping
```

### Set Up Virtual Environment
Create and activate a virtual environment to isolate dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
.\venv\Scripts\activate   # On Windows
```

### Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

For GPU acceleration, ensure you have the appropriate CUDA-compatible packages:
```bash
pip install onnxruntime-gpu
```

### Download Weights File
Download the `inswapper_128.onnx` weights file:
- Access the weights file from [this Google Drive link](https://drive.google.com/file/d/1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF/view?usp=sharing)
- Place the `inswapper_128.onnx` file in the `weights/` directory
- Update the pipeline configuration to reference the correct path if needed

## Usage

### Start the API Server
Launch the FastAPI server:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### Face Swap Endpoint
- **URL**: `/face-swap`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `multi_face_img`: Image file containing multiple faces (target image)
  - `single_face_img`: Image file containing a single face (source image)
- **Response**: Returns the processed image with swapped faces

#### API Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Testing the API

#### Using cURL
```bash
curl -X POST "http://localhost:8000/face-swap" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "multi_face_img=@target_image.jpg" \
  -F "single_face_img=@source_image.jpg" \
  --output result.jpg
```

#### Using Python requests
```python
import requests

url = "http://localhost:8000/face-swap"
files = {
    'multi_face_img': open('target_image.jpg', 'rb'),
    'single_face_img': open('source_image.jpg', 'rb')
}

response = requests.post(url, files=files)

if response.status_code == 200:
    with open('result.jpg', 'wb') as f:
        f.write(response.content)
    print("Face swap completed successfully!")
else:
    print(f"Error: {response.status_code}")
```

#### Using JavaScript/Fetch
```javascript
const formData = new FormData();
formData.append('multi_face_img', multiFileInput.files[0]);
formData.append('single_face_img', singleFileInput.files[0]);

fetch('http://localhost:8000/face-swap', {
    method: 'POST',
    body: formData
})
.then(response => response.blob())
.then(blob => {
    const url = window.URL.createObjectURL(blob);
    const img = document.createElement('img');
    img.src = url;
    document.body.appendChild(img);
});
```

## Code Structure

| Component | Description |
|-----------|-------------|
| `app.py` | FastAPI application with CORS middleware and face swap endpoint |
| `src/pipeline/faceswap_pipeline.py` | Core face swapping pipeline implementation |
| `src/logger/` | Logging configuration and utilities |
| `src/exceptions/` | Custom exception handling |
| `src/components/` | Reusable components for face detection and processing |
| `src/constants/` | Application constants and configuration |
| `src/entity/` | Data models and entity definitions |
| `src/utils/` | Utility functions and helpers |
| `uploads/` | Temporary storage for uploaded images |
| `weights/` | Directory for model weights files |

### Main Application Flow
1. **File Upload**: Images are uploaded via multipart/form-data
2. **File Processing**: Files are saved with unique identifiers in the uploads directory
3. **Pipeline Execution**: The face swap pipeline processes the images
4. **Result Return**: Processed image is returned as a file response
5. **Cleanup**: Temporary files are managed automatically

## Customization

### Adjust Detection Thresholds
Modify detection parameters in the pipeline configuration:
```python
# In src/pipeline/faceswap_pipeline.py
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)  # Lower det_thresh to detect more faces
```

### Configure CORS Settings
Update CORS settings in `app.py` for production use:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Replace with specific domain(s)
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### Modify Upload Directory
Change the upload directory location:
```python
UPLOAD_DIR = "custom_uploads"  # Update in app.py
os.makedirs(UPLOAD_DIR, exist_ok=True)
```

### Environment Configuration
Create a `.env` file for environment-specific settings:
```env
UPLOAD_DIR=uploads
MODEL_PATH=weights/inswapper_128.onnx
LOG_LEVEL=INFO
MAX_FILE_SIZE=10485760  # 10MB
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No module named 'fastapi'` | Install FastAPI: `pip install fastapi uvicorn` |
| `No module named 'onnxruntime'` | Install the missing package: `pip install onnxruntime` or `onnxruntime-gpu` |
| `No faces detected` | Ensure images are clear and well-lit. Lower `det_thresh` or use higher-resolution images |
| `CUDA out of memory` | Reduce `det_size` (e.g., `(320, 320)`) or switch to CPU mode (`ctx_id=-1`) |
| `inswapper_128.onnx not found` | Verify the weights file is in the `weights/` directory |
| `CORS errors` | Update CORS settings in `app.py` to include your frontend domain |
| `File too large` | Check file size limits and adjust server configuration |
| `Server won't start` | Ensure port 8000 is available or use a different port: `--port 8001` |

### Common API Errors
- **400 Bad Request**: Check that both image files are provided
- **500 Internal Server Error**: Check logs for detailed error information
- **422 Unprocessable Entity**: Ensure files are valid image formats

## Development

### Running in Development Mode
```bash
uvicorn app:app --reload --log-level debug
```

### Testing with Jupyter Notebook
Use the provided `insightface.ipynb` notebook for testing and experimentation.

### Logging
Logs are stored in the `logs/` directory. Configure log levels in the logger module.

## Production Deployment

### Using Docker
Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Using Gunicorn
```bash
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## License
This project is licensed under the MIT License, permitting both personal and commercial use.

## Contact
For questions or issues, open an issue on the GitHub repository or contact `your.email@example.com`.

---

### Notes for Developers
1. Ensure the `weights/inswapper_128.onnx` file is present before running the application
2. The `uploads/` directory will be created automatically if it doesn't exist
3. For production deployment, configure appropriate CORS origins and file size limits
4. Monitor the `logs/` directory for application logs and debugging information
5. The pipeline module can be extended to support additional face processing features