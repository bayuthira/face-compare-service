import secrets
from typing import Annotated

import cv2 as cv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.face_service import FaceCompareService, FaceServiceError

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="CPU-only face verification service using OpenCV YuNet and SFace.",
)

face_service = FaceCompareService(settings)


def verify_auth_code(x_auth_code: str | None) -> None:
    expected_auth_code = settings.face_api_auth_code

    # Kalau FACE_API_AUTH_CODE kosong, auth dimatikan.
    # Untuk production sebaiknya jangan kosong.
    if expected_auth_code == "":
        return

    if x_auth_code is None or x_auth_code == "":
        raise HTTPException(
            status_code=401,
            detail={
                "error": "AUTH_CODE_REQUIRED",
                "message": "Header X-Auth-Code wajib dikirim.",
            },
        )

    if not secrets.compare_digest(x_auth_code, expected_auth_code):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "INVALID_AUTH_CODE",
                "message": "Auth code tidak valid.",
            },
        )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.app_name,
        "opencv_version": cv.__version__,
        "threshold": settings.face_threshold,
        "opencv_threads": settings.opencv_threads,
        "auth_enabled": settings.face_api_auth_code != "",
    }


@app.post("/verify")
async def verify(
    reference_image: Annotated[UploadFile, File(description="Foto referensi karyawan")],
    probe_image: Annotated[UploadFile, File(description="Foto absensi terbaru")],
    threshold: Annotated[float | None, Form(description="Optional threshold override")] = None,
    x_auth_code: Annotated[str | None, Header(alias="X-Auth-Code")] = None,
):
    verify_auth_code(x_auth_code)

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
