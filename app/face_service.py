import os
import threading
from dataclasses import dataclass
from typing import Any

import cv2 as cv
import numpy as np

from app.config import Settings


class FaceServiceError(Exception):
    status_code = 400
    error_code = "FACE_SERVICE_ERROR"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class InvalidImageError(FaceServiceError):
    error_code = "INVALID_IMAGE"


class NoFaceError(FaceServiceError):
    error_code = "NO_FACE_DETECTED"


class MultipleFacesError(FaceServiceError):
    error_code = "MULTIPLE_FACES_DETECTED"


class ModelNotFoundError(FaceServiceError):
    status_code = 500
    error_code = "MODEL_NOT_FOUND"


class InvalidEmbeddingError(FaceServiceError):
    error_code = "INVALID_EMBEDDING"


@dataclass
class FaceInfo:
    face_count: int
    box: list[float]
    score: float


@dataclass
class CompareResult:
    match: bool
    similarity: float
    threshold: float
    distance_l2: float
    reference_face: FaceInfo | None
    probe_face: FaceInfo
    probe_embedding: list[float] | None = None


class FaceCompareService:
    def __init__(self, settings: Settings):
        self.settings = settings

        self._validate_models()

        cv.setNumThreads(settings.opencv_threads)

        self.detector = cv.FaceDetectorYN.create(
            settings.face_detect_model,
            "",
            (320, 320),
            settings.detect_score_threshold,
            settings.detect_nms_threshold,
            settings.detect_top_k,
            cv.dnn.DNN_BACKEND_OPENCV,
            cv.dnn.DNN_TARGET_CPU,
        )

        self.recognizer = cv.FaceRecognizerSF.create(
            settings.face_recognition_model,
            "",
            cv.dnn.DNN_BACKEND_OPENCV,
            cv.dnn.DNN_TARGET_CPU,
        )

        # FaceDetectorYN.setInputSize() mengubah state detector.
        # Supaya aman saat request paralel, kita lock bagian OpenCV.
        self._lock = threading.Lock()

    def _validate_models(self) -> None:
        missing = []

        if not os.path.exists(self.settings.face_detect_model):
            missing.append(self.settings.face_detect_model)

        if not os.path.exists(self.settings.face_recognition_model):
            missing.append(self.settings.face_recognition_model)

        if missing:
            raise ModelNotFoundError(
                "Model file tidak ditemukan: " + ", ".join(missing)
            )

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        if not image_bytes:
            raise InvalidImageError("File gambar kosong.")

        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        if len(image_bytes) > max_bytes:
            raise InvalidImageError(
                f"Ukuran gambar terlalu besar. Maksimal {self.settings.max_upload_mb} MB per file."
            )

        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv.imdecode(buffer, cv.IMREAD_COLOR)

        if image is None:
            raise InvalidImageError(
                "File tidak bisa dibaca sebagai gambar. Gunakan JPG/PNG yang valid."
            )

        return image

    def _detect_one_face(self, image: np.ndarray) -> tuple[np.ndarray, FaceInfo]:
        height, width = image.shape[:2]

        self.detector.setInputSize((width, height))

        _, faces = self.detector.detect(image)

        if faces is None or len(faces) == 0:
            raise NoFaceError("Tidak ada wajah yang terdeteksi pada gambar.")

        face_count = len(faces)

        if face_count > 1 and not self.settings.allow_multiple_faces:
            raise MultipleFacesError(
                f"Terdeteksi {face_count} wajah. Untuk absensi, kirim foto dengan satu wajah saja."
            )

        # Jika ALLOW_MULTIPLE_FACES=true, ambil wajah terbesar.
        # Format face: x, y, w, h, landmarks..., score
        face = max(faces, key=lambda f: float(f[2] * f[3]))

        box = [
            float(face[0]),
            float(face[1]),
            float(face[2]),
            float(face[3]),
        ]

        score = float(face[14])

        return face, FaceInfo(
            face_count=face_count,
            box=box,
            score=score,
        )

    def _extract_feature(self, image: np.ndarray) -> tuple[np.ndarray, FaceInfo]:
        face, face_info = self._detect_one_face(image)

        aligned_face = self.recognizer.alignCrop(image, face)

        feature = self.recognizer.feature(aligned_face)

        return feature, face_info

    def _normalize_embedding(self, embedding: list[float]) -> np.ndarray:
        if not embedding:
            raise InvalidEmbeddingError("Field reference_embedding tidak boleh kosong.")

        try:
            feature = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        except ValueError:
            raise InvalidEmbeddingError(
                "Field reference_embedding harus berupa array angka desimal."
            )

        if feature.size == 0:
            raise InvalidEmbeddingError("Field reference_embedding tidak boleh kosong.")

        if not np.isfinite(feature).all():
            raise InvalidEmbeddingError(
                "Field reference_embedding mengandung nilai tidak valid."
            )

        return feature

    def extract_embedding_bytes(self, image_bytes: bytes) -> tuple[np.ndarray, FaceInfo]:
        image = self._decode_image(image_bytes)

        with self._lock:
            feature, face_info = self._extract_feature(image)

        return feature, face_info

    def compare_bytes(
        self,
        reference_image_bytes: bytes,
        probe_image_bytes: bytes,
        threshold: float | None = None,
    ) -> CompareResult:
        if threshold is None:
            threshold = self.settings.face_threshold

        reference_image = self._decode_image(reference_image_bytes)
        probe_image = self._decode_image(probe_image_bytes)

        with self._lock:
            reference_feature, reference_face = self._extract_feature(reference_image)
            probe_feature, probe_face = self._extract_feature(probe_image)

            similarity = self.recognizer.match(
                reference_feature,
                probe_feature,
                cv.FaceRecognizerSF_FR_COSINE,
            )

            distance_l2 = self.recognizer.match(
                reference_feature,
                probe_feature,
                cv.FaceRecognizerSF_FR_NORM_L2,
            )

        similarity = float(similarity)
        distance_l2 = float(distance_l2)

        return CompareResult(
            match=similarity >= threshold,
            similarity=similarity,
            threshold=float(threshold),
            distance_l2=distance_l2,
            reference_face=reference_face,
            probe_face=probe_face,
            probe_embedding=self.embedding_to_list(probe_feature),
        )

    def compare_embedding_with_probe_bytes(
        self,
        reference_embedding: list[float],
        probe_image_bytes: bytes,
        threshold: float | None = None,
    ) -> CompareResult:
        if threshold is None:
            threshold = self.settings.face_threshold

        reference_feature = self._normalize_embedding(reference_embedding)
        probe_image = self._decode_image(probe_image_bytes)

        try:
            with self._lock:
                probe_feature, probe_face = self._extract_feature(probe_image)

                similarity = self.recognizer.match(
                    reference_feature,
                    probe_feature,
                    cv.FaceRecognizerSF_FR_COSINE,
                )

                distance_l2 = self.recognizer.match(
                    reference_feature,
                    probe_feature,
                    cv.FaceRecognizerSF_FR_NORM_L2,
                )
        except cv.error:
            raise InvalidEmbeddingError(
                "Format reference_embedding tidak valid atau dimensi tidak sesuai model."
            )

        similarity = float(similarity)
        distance_l2 = float(distance_l2)

        return CompareResult(
            match=similarity >= threshold,
            similarity=similarity,
            threshold=float(threshold),
            distance_l2=distance_l2,
            reference_face=None,
            probe_face=probe_face,
            probe_embedding=self.embedding_to_list(probe_feature),
        )

    @staticmethod
    def embedding_to_list(feature: np.ndarray) -> list[float]:
        return [round(float(value), 8) for value in feature.flatten()]

    @staticmethod
    def result_to_dict(
        result: CompareResult,
        include_probe_embedding: bool = False,
    ) -> dict[str, Any]:
        probe_face_payload: dict[str, Any] = {
            "face_count": result.probe_face.face_count,
            "box": result.probe_face.box,
            "score": round(result.probe_face.score, 6),
        }

        if include_probe_embedding and result.probe_embedding is not None:
            probe_face_payload["embedding"] = result.probe_embedding

        payload = {
            "match": result.match,
            "similarity": round(result.similarity, 6),
            "threshold": result.threshold,
            "distance_l2": round(result.distance_l2, 6),
            "message": "same person" if result.match else "different person",
            "probe_face": probe_face_payload,
        }

        if result.reference_face is not None:
            payload["reference_face"] = {
                "face_count": result.reference_face.face_count,
                "box": result.reference_face.box,
                "score": round(result.reference_face.score, 6),
            }

        return payload
