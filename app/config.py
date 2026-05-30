import os
from dataclasses import dataclass


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default

    return value.lower() in ("1", "true", "yes", "y", "on")


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "face-compare-service")

    face_detect_model: str = os.getenv(
        "FACE_DETECT_MODEL",
        "/app/models/face_detection_yunet_2023mar.onnx",
    )

    face_recognition_model: str = os.getenv(
        "FACE_RECOGNITION_MODEL",
        "/app/models/face_recognition_sface_2021dec.onnx",
    )

    face_threshold: float = _get_float("FACE_THRESHOLD", 0.363)

    detect_score_threshold: float = _get_float("DETECT_SCORE_THRESHOLD", 0.90)
    detect_nms_threshold: float = _get_float("DETECT_NMS_THRESHOLD", 0.30)
    detect_top_k: int = _get_int("DETECT_TOP_K", 5000)

    max_upload_mb: int = _get_int("MAX_UPLOAD_MB", 8)

    allow_multiple_faces: bool = _get_bool("ALLOW_MULTIPLE_FACES", False)

    opencv_threads: int = _get_int("OPENCV_THREADS", 1)

    # Auth code untuk endpoint /verify.
    # Jika kosong, endpoint /verify tidak butuh auth.
    # Untuk production, wajib isi ini di .env.
    face_api_auth_code: str = os.getenv("FACE_API_AUTH_CODE", "")


settings = Settings()
