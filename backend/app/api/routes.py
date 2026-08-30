"""HTTP routes.

Deliberately thin: parse the request, delegate to services, shape the response.
No engine-specific logic belongs in this file.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, File, UploadFile
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.dependencies import get_asr_service, get_repair_service, get_tts_service
from app.models.schemas import HealthResponse, ProcessSpeechResponse
from app.services import audio as audio_service
from app.services import uncertainty
from app.services.asr import ASRService
from app.services.repair import RepairService
from app.services.tts import TTSService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Liveness probe. Also handy for checking LAN reachability from a phone."""
    return HealthResponse()


@router.post(
    "/process-speech",
    response_model=ProcessSpeechResponse,
    tags=["speech"],
    summary="Transcribe an audio recording",
)
async def process_speech(
    audio: UploadFile = File(..., description="Recorded audio. Any common container."),
    settings: Settings = Depends(get_settings),
    asr: ASRService = Depends(get_asr_service),
    repairer: RepairService = Depends(get_repair_service),
    tts: TTSService = Depends(get_tts_service),
) -> ProcessSpeechResponse:
    """Audio in, clarified sentence out.

    `repaired_text` is a tidy pass-through until the repair model lands -- see
    the milestones in README.md.
    """
    started = time.perf_counter()
    temp_files: list = []

    logger.info(
        "POST /process-speech file=%s content_type=%s",
        audio.filename, audio.content_type,
    )

    try:
        source = await audio_service.save_upload_to_temp(
            audio,
            max_bytes=settings.max_upload_bytes,
            temp_dir=settings.temp_dir,
        )
        audio_path, temp_files = audio_service.prepare_audio(
            source,
            temp_dir=settings.temp_dir,
            use_ffmpeg=settings.transcode_with_ffmpeg,
        )

        duration = await run_in_threadpool(audio_service.probe_audio, audio_path)
        if duration is not None and duration < settings.min_audio_seconds:
            logger.info("retry: recording is only %.2fs of audio", duration)
            return ProcessSpeechResponse(status="retry")

        # Whisper inference is blocking and CPU-bound; keep the event loop free.
        result = await run_in_threadpool(
            asr.transcribe, audio_path, language=settings.language
        )

        # Constrained selector repair needs the audio (for the N-best pass),
        # so it runs before cleanup. `select` returns None whenever it cannot
        # or should not run, and the legacy path below is used unchanged.
        selection = None
        if hasattr(repairer, "select"):
            selection = await run_in_threadpool(
                repairer.select, audio_path, result, asr
            )
    finally:
        audio_service.cleanup(*temp_files)

    uncertain_word = uncertainty.find_uncertain_word(result, settings)
    status = uncertainty.decide_status(result, uncertain_word, settings)

    if status == "retry":
        logger.info(
            "retry: nothing usable heard (chars=%d, confidence=%s) in %.0f ms",
            len(result.text), _fmt(result.confidence), _elapsed_ms(started),
        )
        return ProcessSpeechResponse(
            status="retry",
            raw_transcript=result.text.strip() or None,
            confidence=result.confidence,
        )

    if selection is not None:
        # Runtime invariant: the selector's output is always a member of its
        # own hypothesis list; `decide()` enforces it and falls back to A0.
        repaired = selection.final_display
        # KEEP_A0 says nothing about ASR confidence — it only means the
        # selector chose not to alter A0. The legacy confidence-driven status
        # therefore stands for KEEP_A0; only UNCERTAIN overrides it.
        if selection.decision == "UNCERTAIN":
            status = "uncertain"
        logger.debug(
            "selector: %s margin=%s candidates=%d nbest=%.0fms select=%.0fms",
            selection.decision, selection.margin, selection.n_candidates,
            selection.nbest_ms or -1, selection.selector_ms or -1,
        )
    else:
        repaired = repairer.repair(result)

    # Only speak a sentence we are actually going to hand over. An uncertain
    # sentence still has a hole in it, so there is nothing to synthesise yet.
    audio_url = None
    if status == "success":
        spoken = repaired or result.text
        audio_url = await run_in_threadpool(tts.synthesize, spoken)

    logger.info(
        "%s: backend=%s chars=%d confidence=%s audio=%s in %.0f ms",
        status, asr.name, len(result.text), _fmt(result.confidence),
        audio_url or "none", _elapsed_ms(started),
    )

    if selection is not None:
        from app.models.schemas import RepairAlternative

        # Contract bands. KEEP_A0 sends decision=null so the frontend keeps
        # deriving its band from the EXISTING status + ASR-confidence mapping —
        # the selector's agreement is not an ASR-confidence claim. UNCERTAIN
        # never auto-speaks: a strong suggestion maps to "medium" (the
        # confident "I think you said..." one-tap flow), a weak one to "low".
        if selection.decision == "UNCERTAIN":
            decision_band = ("medium" if selection.suggestion_strength == "strong"
                             else "low")
        else:
            decision_band = None
        return ProcessSpeechResponse(
            status=status,
            raw_transcript=result.text,
            repaired_text=repaired,
            confidence=result.confidence,
            uncertain_words=([uncertain_word] if uncertain_word
                             and selection.decision != "UNCERTAIN" else []),
            audio_url=audio_url,
            repair_available=True,
            decision=decision_band,
            repair_decision=selection.decision,
            suggested_text=selection.suggested_text,
            alternatives=[RepairAlternative(text=t) for t in selection.alternatives],
        )

    return ProcessSpeechResponse(
        status=status,
        raw_transcript=result.text,
        repaired_text=repaired,
        confidence=result.confidence,
        uncertain_words=[uncertain_word] if uncertain_word else [],
        audio_url=audio_url,
    )


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _fmt(confidence: float | None) -> str:
    return "n/a" if confidence is None else f"{confidence:.2f}"
