"""faster-whisper configuration for this project.

This project is deliberately standalone: it does NOT import anything from
``~/Desktop/speech-repair/``. The values below were copied by hand so that
offline evaluation runs the *same* ASR configuration as the Revoice backend
(``backend/app/config.py`` and ``backend/app/services/asr.py``).

If the backend's settings change, update this file to match -- there is no
automatic link between the two projects.

Every value can be overridden with an environment variable of the same name,
using the same names the backend uses (WHISPER_MODEL, WHISPER_DEVICE, ...).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


@dataclass(frozen=True)
class ASRConfig:
    """Mirrors the Revoice backend's faster-whisper settings."""

    # --- model construction (WhisperModel(...)) ---------------------------
    # Backend default: WHISPER_MODEL=base.en
    model_size: str = field(default_factory=lambda: _env("WHISPER_MODEL", "base.en"))
    # Backend default: WHISPER_DEVICE=auto (Apple Silicon resolves to CPU;
    # CTranslate2 has no Metal backend).
    device: str = field(default_factory=lambda: _env("WHISPER_DEVICE", "auto"))
    # Backend default: WHISPER_COMPUTE_TYPE=int8
    compute_type: str = field(default_factory=lambda: _env("WHISPER_COMPUTE_TYPE", "int8"))

    # --- decoding (model.transcribe(...)) ---------------------------------
    # Backend default: WHISPER_LANGUAGE=en
    language: str | None = field(default_factory=lambda: _env("WHISPER_LANGUAGE", "en") or None)
    beam_size: int = 5
    vad_filter: bool = True
    condition_on_previous_text: bool = False
    word_timestamps: bool = True

    def transcribe_kwargs(self) -> dict[str, Any]:
        """Exactly the keyword arguments the backend passes to transcribe()."""
        return {
            "language": self.language,
            "beam_size": self.beam_size,
            "vad_filter": self.vad_filter,
            "condition_on_previous_text": self.condition_on_previous_text,
            "word_timestamps": self.word_timestamps,
        }

    def describe(self) -> dict[str, Any]:
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            **self.transcribe_kwargs(),
        }


#: Sample rate faster-whisper expects. The backend hands it a file and lets
#: PyAV resample; here we resample ourselves so the model always sees 16 kHz.
TARGET_SAMPLE_RATE = 16_000

#: Hugging Face dataset under evaluation.
HF_DATASET_ID = "resproj007/torgo_dysarthric_male"
