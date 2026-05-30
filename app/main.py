import secrets

import cv2 as cv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.face_service import FaceCompareService, FaceServiceError

app = FastAPI(
    title=settings.app_name,
    version="1.0.2",
    description="CPU-only face verification service using OpenCV YuNet and SFace.",
)

face_service = FaceCompareService(settings)


def verify_auth_code(x_auth_code: str | None) -> None:
    expected_auth_code = settings.face_api_auth_code

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


def validate_request_has_body(request: Request) -> None:
    content_length = request.headers.get("content-length")
    content_type = request.headers.get("content-type", "")

    if content_length is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "BODY_REQUIRED",
                "message": "Request body wajib dikirim dalam format multipart/form-data.",
            },
        )

    try:
        length = int(content_length)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "INVALID_CONTENT_LENGTH",
                "message": "Header Content-Length tidak valid.",
            },
        )

    if length <= 0:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "BODY_REQUIRED",
                "message": "Request body kosong. Field reference_image dan probe_image wajib dikirim.",
            },
        )

    if "multipart/form-data" not in content_type.lower():
        raise HTTPException(
            status_code=415,
            detail={
                "error": "UNSUPPORTED_MEDIA_TYPE",
                "message": "Content-Type harus multipart/form-data.",
            },
        )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": "1.0.2",
        "opencv_version": cv.__version__,
        "threshold": settings.face_threshold,
        "opencv_threads": settings.opencv_threads,
        "auth_enabled": settings.face_api_auth_code != "",
    }


@app.post("/verify")
async def verify(
    request: Request,
    x_auth_code: str | None = Header(default=None, alias="X-Auth-Code"),
):
    verify_auth_code(x_auth_code)

    validate_request_has_body(request)

    try:
        form = await request.form()

        reference_image = form.get("reference_image")
        probe_image = form.get("probe_image")
        threshold_value = form.get("threshold")

        if reference_image is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "REFERENCE_IMAGE_REQUIRED",
                    "message": "Field reference_image wajib dikirim.",
                },
            )

        if probe_image is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "PROBE_IMAGE_REQUIRED",
                    "message": "Field probe_image wajib dikirim.",
                },
            )

        if not hasattr(reference_image, "read"):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "REFERENCE_IMAGE_INVALID",
                    "message": "Field reference_image harus berupa file.",
                },
            )

        if not hasattr(probe_image, "read"):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "PROBE_IMAGE_INVALID",
                    "message": "Field probe_image harus berupa file.",
                },
            )

        threshold = None
        if threshold_value not in (None, ""):
            try:
                threshold = float(str(threshold_value))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "THRESHOLD_INVALID",
                        "message": "Field threshold harus berupa angka, contoh: 0.363.",
                    },
                )

        reference_bytes = await reference_image.read()
        probe_bytes = await probe_image.read()

        if len(reference_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "REFERENCE_IMAGE_EMPTY",
                    "message": "File reference_image kosong.",
                },
            )

        if len(probe_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "PROBE_IMAGE_EMPTY",
                    "message": "File probe_image kosong.",
                },
            )

        result = face_service.compare_bytes(
            reference_image_bytes=reference_bytes,
            probe_image_bytes=probe_bytes,
            threshold=threshold,
        )

        return face_service.result_to_dict(result)

    except HTTPException:
        raise

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
