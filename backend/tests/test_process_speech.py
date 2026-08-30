import shutil

import pytest
from fastapi.testclient import TestClient


def post_audio(client: TestClient, wav_bytes: bytes, name: str = "recording.wav") -> dict:
    response = client.post(
        "/process-speech",
        files={"audio": (name, wav_bytes, "audio/wav")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_returns_the_documented_contract(client: TestClient, wav_bytes: bytes) -> None:
    body = post_audio(client, wav_bytes)

    assert body["status"] == "success"
    assert body["raw_transcript"] == "could you get me some water"
    # Placeholder repair: tidied, never reworded.
    assert body["repaired_text"] == "Could you get me some water?"
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["uncertain_words"] == []
    assert body["audio_url"] is None  # TTS_BACKEND=none in tests
    # Contract extended additively for the constrained-selector repair
    # (API_CONTRACT.md): repair_available/decision/repair_decision/
    # suggested_text/alternatives. Legacy fields are unchanged.
    assert set(body) == {
        "status",
        "raw_transcript",
        "repaired_text",
        "confidence",
        "uncertain_words",
        "audio_url",
        "repair_available",
        "decision",
        "repair_decision",
        "suggested_text",
        "alternatives",
    }
    assert body["repair_available"] is False  # passthrough repair, no selector


def test_low_word_confidence_becomes_uncertain(
    client: TestClient, wav_bytes: bytes, monkeypatch
) -> None:
    from app.config import get_settings
    from app.dependencies import reset_asr_service

    monkeypatch.setenv("MOCK_WORD_PROBABILITY", "0.2")
    get_settings.cache_clear()
    reset_asr_service()

    body = post_audio(client, wav_bytes)

    assert body["status"] == "uncertain"
    word = body["uncertain_words"][0]
    # The frontend blanks out `position` in `raw_transcript.split()`.
    assert body["raw_transcript"].split()[word["position"]] == word["original"]
    assert word["options"]


def test_empty_transcript_becomes_retry(
    client: TestClient, wav_bytes: bytes, monkeypatch
) -> None:
    from app.services.asr import MockASRService, TranscriptionResult

    monkeypatch.setattr(
        MockASRService,
        "transcribe",
        lambda self, path, language=None: TranscriptionResult(text="", language="en"),
    )

    body = post_audio(client, wav_bytes)

    assert body["status"] == "retry"
    assert body["raw_transcript"] is None
    assert body["audio_url"] is None


@pytest.mark.skipif(shutil.which("say") is None, reason="system TTS needs macOS `say`")
def test_system_tts_returns_a_playable_url(
    client: TestClient, wav_bytes: bytes, tmp_path, monkeypatch
) -> None:
    from app.config import get_settings
    from app.dependencies import reset_asr_service
    from app.main import create_app

    monkeypatch.setenv("TTS_BACKEND", "system")
    monkeypatch.setenv("AUDIO_OUTPUT_DIR", str(tmp_path))
    get_settings.cache_clear()
    reset_asr_service()

    # A fresh app so the /audio mount points at tmp_path.
    with TestClient(create_app()) as tts_client:
        body = post_audio(tts_client, wav_bytes)

        assert body["audio_url"].startswith("/audio/")
        served = tts_client.get(body["audio_url"])
        assert served.status_code == 200
        assert len(served.content) > 1000


def test_accepts_m4a_from_expo(client: TestClient, wav_bytes: bytes) -> None:
    # expo-audio writes .m4a on both iOS and Android by default.
    response = client.post(
        "/process-speech",
        files={"audio": ("recording.m4a", wav_bytes, "audio/m4a")},
    )
    assert response.status_code == 200
    assert response.json()["raw_transcript"]


def test_missing_field_is_a_validation_error(client: TestClient) -> None:
    response = client.post("/process-speech")
    assert response.status_code == 422
    body = response.json()
    assert body["status"] == "error"
    assert body["error"]["code"] == "validation_error"


def test_empty_file_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/process-speech",
        files={"audio": ("recording.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_audio"


def test_unsupported_extension_is_rejected(client: TestClient, wav_bytes: bytes) -> None:
    response = client.post(
        "/process-speech",
        files={"audio": ("notes.txt", wav_bytes, "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_audio"


def test_oversized_upload_is_rejected(client: TestClient, monkeypatch) -> None:
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1024")
    get_settings.cache_clear()

    response = client.post(
        "/process-speech",
        files={"audio": ("big.wav", b"\x00" * 4096, "audio/wav")},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "audio_too_large"


def test_a_mis_tap_recording_is_a_retry(client: TestClient) -> None:
    """Stopping a recording immediately is "I didn't catch that", not an error."""
    import io
    import struct
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(struct.pack("<800h", *([0] * 800)))  # 0.05 s

    body = post_audio(client, buffer.getvalue())
    assert body["status"] == "retry"
    assert body["raw_transcript"] is None


def test_undecodable_audio_is_a_422_not_a_500(client: TestClient) -> None:
    # What a browser produces when a recording is stopped before any frame is
    # written: the right extension, no readable audio inside.
    response = client.post(
        "/process-speech",
        files={"audio": ("recording.webm", b"\x1aE\xdf\xa3" + b"\x00" * 106, "audio/webm")},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "audio_decode_failed"


def test_temp_files_are_cleaned_up(client: TestClient, wav_bytes: bytes, tmp_path, monkeypatch) -> None:
    from app.config import get_settings

    monkeypatch.setenv("TEMP_DIR", str(tmp_path))
    get_settings.cache_clear()

    response = client.post(
        "/process-speech",
        files={"audio": ("recording.wav", wav_bytes, "audio/wav")},
    )
    assert response.status_code == 200
    assert list(tmp_path.iterdir()) == []
