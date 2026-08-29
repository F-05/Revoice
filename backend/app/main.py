"""FastAPI application entrypoint.

Run with:  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import get_settings
from app.dependencies import get_asr_service, reset_asr_service
from app.errors import register_exception_handlers
from app.services.audio import ffmpeg_path
from app.services.tts import AUDIO_URL_PREFIX, resolve_audio_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    logger.info("ASR backend: %s (model=%s)", settings.asr_backend, settings.whisper_model)
    logger.info("TTS backend: %s", settings.tts_backend)
    logger.info("CORS origins: %s", settings.cors_origins)

    if ffmpeg_path() is None:
        logger.warning(
            "ffmpeg not found on PATH. The faster-whisper backend decodes audio "
            "itself so this is fine; `brew install ffmpeg` if you switch to the "
            "openai-whisper backend or hit a container it cannot read."
        )

    if settings.preload_asr_model:
        # Load weights now so the first real request is not slow.
        try:
            get_asr_service(settings).load()
        except Exception:
            logger.exception("ASR model preload failed; will retry on first request.")

    yield

    reset_asr_service()
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Speech capture -> ASR -> (repair, TTS to come).",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        # Must stay False while allow_origins can be "*".
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    app.include_router(router)

    # Synthesised speech is served from here; `audio_url` in the response is a
    # path under this mount, which the frontend resolves against its API base.
    audio_dir = resolve_audio_dir(settings)
    audio_dir.mkdir(parents=True, exist_ok=True)
    app.mount(AUDIO_URL_PREFIX, StaticFiles(directory=audio_dir), name="audio")

    return app


app = create_app()
