from dataclasses import dataclass

@dataclass
class ModelInitializationArtifact:
    model_name: str

@dataclass
class SwapperModelArtifact:
    result_image_path: str
    detected_face_paths: list
    base64_faces: list