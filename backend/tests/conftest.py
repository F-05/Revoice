from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _mock_asr_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the whole suite against the mock ASR backend.

    Tests exercise the API contract and the audio plumbing, not Whisper's
    accuracy -- that belongs in evaluation/.
    """
    monkeypatch.setenv("ASR_BACKEND", "mock")
    monkeypatch.setenv("PRELOAD_ASR_MODEL", "false")
    # Skip the ffmpeg round-trip so tests do not depend on a system binary.
    monkeypatch.setenv("TRANSCODE_WITH_FFMPEG", "false")
    # ...and no `say` round-trip either. The TTS path has its own test.
    monkeypatch.setenv("TTS_BACKEND", "none")

    from app.config import get_settings
    from app.dependencies import reset_asr_service

    get_settings.cache_clear()
    reset_asr_service()
    yield
    get_settings.cache_clear()
    reset_asr_service()


@pytest.fixture
def client() -> TestClient:
    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def wav_bytes() -> bytes:
    """A tiny valid 16 kHz mono WAV (0.5 s of silence).

    Comfortably over `min_audio_seconds`, so it reaches the ASR backend.
    """
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(struct.pack("<8000h", *([0] * 8000)))
    return buffer.getvalue()
