"""Application error types and their HTTP representation."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.models.schemas import ErrorDetail, ErrorResponse


class AppError(Exception):
    """An error we can describe to the client with a stable machine code."""

    status_code: int = 400
    code: str = "bad_request"

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class InvalidAudioError(AppError):
    status_code = 400
    code = "invalid_audio"


class AudioTooLargeError(AppError):
    status_code = 413
    code = "audio_too_large"


class AudioDecodeError(AppError):
    status_code = 422
    code = "audio_decode_failed"


class ASRUnavailableError(AppError):
    status_code = 503
    code = "asr_unavailable"


class TranscriptionError(AppError):
    status_code = 500
    code = "transcription_failed"


def _error_response(status_code: int, code: str, message: str, detail: str | None = None) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(code=code, message=message, detail=detail))
    return JSONResponse(status_code=status_code, content=body.model_dump())


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return _error_response(exc.status_code, exc.code, exc.message, exc.detail)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(
            422,
            "validation_error",
            "The request body was not valid.",
            str(exc.errors()),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        return _error_response(
            500,
            "internal_error",
            "Something went wrong while processing the request.",
            f"{type(exc).__name__}: {exc}",
        )
