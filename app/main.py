from typing import Annotated

import cv2 as cv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.face_service import FaceCompareService, FaceServiceError

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="CPU-only face verification service using OpenCV YuNet and SFace.",
)

face_service = FaceCompareService(settings)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.app_name,
        "opencv_version": cv.__version__,
        "threshold": settings.face_threshold,
        "opencv_threads": settings.opencv_threads,
    }


@app.post("/verify")
async def verify(
    reference_image: Annotated[UploadFile, File(description="Foto referensi karyawan")],
    probe_image: Annotated[UploadFile, File(description="Foto absensi terbaru")],
    threshold: Annotated[float | None, Form(description="Optional threshold override")] = None,
):
    try:
        reference_bytes = await reference_image.read()
        probe_bytes = await probe_image.read()

        result = face_service.compare_bytes(
            reference_image_bytes=reference_bytes,
            probe_image_bytes=probe_bytes,
            threshold=threshold,
        )

        return face_service.result_to_dict(result)

    except FaceServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={
                "error": exc.error_code,
                "message": exc.message,
            },
        )

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_SERVER_ERROR",
                "message": str(exc),
            },
        )
