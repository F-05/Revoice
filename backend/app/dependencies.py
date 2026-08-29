"""Shared FastAPI dependencies.

The ASR service is a process-wide singleton: the model is expensive to load and
is reused across requests. Repair and TTS are cheap, but are cached the same way
so a request never rebuilds them.
"""

from __future__ import annotations

import threading

from fastapi import Depends

from app.config import Settings, get_settings
from app.services.asr import ASRService, build_asr_service
from app.services.repair import RepairService, build_repair_service
from app.services.tts import TTSService, build_tts_service

_asr_service: ASRService | None = None
_asr_lock = threading.Lock()

_repair_service: RepairService | None = None
_tts_service: TTSService | None = None
_service_lock = threading.Lock()


def get_asr_service(settings: Settings = Depends(get_settings)) -> ASRService:
    global _asr_service
    if _asr_service is None:
        with _asr_lock:
            if _asr_service is None:
                _asr_service = build_asr_service(settings)
    return _asr_service


def get_repair_service(settings: Settings = Depends(get_settings)) -> RepairService:
    global _repair_service
    if _repair_service is None:
        with _service_lock:
            if _repair_service is None:
                _repair_service = build_repair_service(settings)
    return _repair_service


def get_tts_service(settings: Settings = Depends(get_settings)) -> TTSService:
    global _tts_service
    if _tts_service is None:
        with _service_lock:
            if _tts_service is None:
                _tts_service = build_tts_service(settings)
    return _tts_service


def reset_asr_service() -> None:
    """Drop the cached services. Used by tests and at shutdown."""
    global _asr_service, _repair_service, _tts_service
    with _asr_lock:
        _asr_service = None
    with _service_lock:
        _repair_service = None
        _tts_service = None
