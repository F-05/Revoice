"""Application configuration.

All settings can be overridden with environment variables (or a local `.env`
file). See `.env.example` for the full list.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ASRBackend = Literal["faster-whisper", "openai-whisper", "mock"]
TTSBackend = Literal["none", "system"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- app ---------------------------------------------------------------
    app_name: str = "speech-repair-backend"
    app_version: str = "0.1.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # Comma-separated list. "*" allows any origin, which is what you want for
    # Expo dev (the phone hits the server on a LAN IP that changes network to
    # network). Credentials are never allowed, so "*" is safe here.
    cors_allow_origins: str = "*"

    # --- ASR ---------------------------------------------------------------
    asr_backend: ASRBackend = "faster-whisper"
    # e.g. tiny.en / base.en / small.en / medium.en / large-v3
    whisper_model: str = "base.en"
    # "auto" | "cpu" | "cuda". Apple Silicon runs on CPU (no Metal in CTranslate2).
    whisper_device: str = "auto"
    # "int8" is the fastest sane default on CPU. "float16" for CUDA.
    whisper_compute_type: str = "int8"
    # Force a language ("en") or set to null/empty to auto-detect.
    whisper_language: str | None = "en"
    # Load the model at startup instead of on the first request.
    preload_asr_model: bool = True
    # Fixed per-word probability reported by ASR_BACKEND=mock. Drop it below
    # `uncertain_word_threshold` to make the mock return status="uncertain".
    mock_word_probability: float = 0.95

    # --- repair (constrained hypothesis selector) ---------------------------
    # none        -> A0-only behaviour, exactly as before the selector existed
    # passthrough -> legacy tidy-only repair
    # selector    -> hybrid N-best + frozen D3-NL selector (needs the artifact)
    repair_backend: Literal["none", "passthrough", "selector"] = "passthrough"
    # Path to the versioned selector artifact (revoice_selector_v1.json).
    repair_model_path: str | None = None
    # Operational override for the switching margin. None -> use the value
    # frozen inside the artifact. An active override is logged loudly.
    repair_switch_margin: float | None = None

    # --- response status ----------------------------------------------------
    # Overall confidence below this -> status="retry" (ask for another take).
    retry_confidence_threshold: float = 0.35
    # Any word below this -> status="uncertain" (ask the user about that word).
    uncertain_word_threshold: float = 0.60

    # --- TTS ----------------------------------------------------------------
    # none   -> audio_url is always null; the app speaks with the device voice.
    # system -> macOS `say`, written to `audio_output_dir` and served at /audio.
    tts_backend: TTSBackend = "system"
    # Where synthesised WAVs live. None -> <system temp>/speech-repair-audio.
    audio_output_dir: str | None = None
    # Generated clips are pruned oldest-first past this count.
    audio_keep_files: int = 50

    # --- audio -------------------------------------------------------------
    max_upload_bytes: int = 25 * 1024 * 1024  # 25 MB
    # Recordings shorter than this are answered with status="retry" instead of
    # being sent to Whisper -- a mis-tap, not speech.
    min_audio_seconds: float = 0.25
    # Directory for temporary uploads. None -> system temp dir.
    temp_dir: str | None = None
    # Transcode uploads to 16 kHz mono WAV with ffmpeg when it is available.
    # Not required: the faster-whisper backend decodes compressed audio itself.
    transcode_with_ffmpeg: bool = True

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    @property
    def language(self) -> str | None:
        """Normalise an empty-string language to None (= auto-detect)."""
        if self.whisper_language and self.whisper_language.strip():
            return self.whisper_language.strip()
        return None


@lru_cache
def get_settings() -> Settings:
    return Settings()
