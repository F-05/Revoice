"""Automatic speech recognition, behind a swappable interface.

The API layer only ever talks to `ASRService`. Whisper-specific inference lives
in the concrete implementations below, so a different engine can be dropped in
by adding a subclass and a line in `build_asr_service`.
"""

from __future__ import annotations

import logging
import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings
from app.errors import ASRUnavailableError, TranscriptionError

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WordScore:
    """One recognised word and how sure the engine is about it."""

    word: str
    probability: float
    start: float | None = None
    end: float | None = None


@dataclass(slots=True)
class TranscriptSegment:
    text: str
    start: float
    end: float
    # Mean token log-probability for the segment, when the engine reports one.
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    words: list[WordScore] = field(default_factory=list)


@dataclass(slots=True)
class TranscriptionResult:
    """Engine-agnostic transcription output."""

    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[TranscriptSegment] = field(default_factory=list)

    @property
    def words(self) -> list[WordScore]:
        """Every scored word, in order. Empty when the engine reports none."""
        return [word for segment in self.segments for word in segment.words]

    @property
    def confidence(self) -> float | None:
        """Duration-weighted mean of exp(avg_logprob), or None if unavailable.

        A rough proxy, not a calibrated probability. Surfaced to the client as
        `confidence`, and used to pick the response status.
        """
        scored = [s for s in self.segments if s.avg_logprob is not None]
        if not scored:
            return None
        weights = [max(s.end - s.start, 1e-3) for s in scored]
        total = sum(weights)
        if total <= 0:
            return None
        value = sum(math.exp(s.avg_logprob) * w for s, w in zip(scored, weights)) / total
        return min(max(value, 0.0), 1.0)


class ASRService(ABC):
    """Interface every ASR engine must satisfy."""

    #: Short identifier, surfaced in logs and /health.
    name: str = "asr"

    @abstractmethod
    def load(self) -> None:
        """Load model weights. Safe to call more than once."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    def transcribe(self, audio_path: Path, *, language: str | None = None) -> TranscriptionResult:
        """Transcribe a local audio file. Blocking; call from a worker thread."""

    def describe(self) -> dict[str, object]:
        return {"backend": self.name, "loaded": self.is_loaded}


class WhisperASRService(ASRService):
    """Local Whisper via `faster-whisper` (CTranslate2).

    Decodes compressed audio itself through PyAV, so no ffmpeg binary is needed.
    """

    name = "faster-whisper"

    def __init__(
        self,
        model_size: str = "base.en",
        *,
        device: str = "auto",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._load_lock = threading.Lock()
        # Whisper models are not safe to call concurrently from many threads.
        self._infer_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover - depends on env
                raise ASRUnavailableError(
                    "faster-whisper is not installed.",
                    detail="pip install -r requirements.txt",
                ) from exc

            logger.info(
                "Loading Whisper model %s (device=%s, compute_type=%s)",
                self.model_size, self.device, self.compute_type,
            )
            try:
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            except Exception as exc:
                raise ASRUnavailableError(
                    f"Could not load Whisper model '{self.model_size}'.",
                    detail=f"{type(exc).__name__}: {exc}",
                ) from exc
            logger.info("Whisper model loaded.")

    def transcribe(self, audio_path: Path, *, language: str | None = None) -> TranscriptionResult:
        self.load()
        assert self._model is not None

        try:
            with self._infer_lock:
                segments_iter, info = self._model.transcribe(
                    str(audio_path),
                    language=language,
                    beam_size=5,
                    vad_filter=True,
                    condition_on_previous_text=False,
                    # Per-word probabilities feed the uncertain-word response.
                    word_timestamps=True,
                )
                segments = [
                    TranscriptSegment(
                        text=s.text.strip(),
                        start=float(s.start),
                        end=float(s.end),
                        avg_logprob=getattr(s, "avg_logprob", None),
                        no_speech_prob=getattr(s, "no_speech_prob", None),
                        words=_word_scores(getattr(s, "words", None)),
                    )
                    for s in segments_iter
                ]
        except Exception as exc:
            raise TranscriptionError(
                "Whisper failed to transcribe the audio.",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

        return TranscriptionResult(
            text=" ".join(s.text for s in segments if s.text).strip(),
            language=getattr(info, "language", None),
            duration=getattr(info, "duration", None),
            segments=segments,
        )


def _word_scores(words: object) -> list[WordScore]:
    """Map an engine's word objects onto `WordScore`, skipping anything odd."""
    if not words:
        return []
    scored: list[WordScore] = []
    for word in words:  # type: ignore[union-attr]
        text = str(getattr(word, "word", "") or "").strip()
        probability = getattr(word, "probability", None)
        if not text or probability is None:
            continue
        scored.append(
            WordScore(
                word=text,
                probability=float(probability),
                start=_maybe_float(getattr(word, "start", None)),
                end=_maybe_float(getattr(word, "end", None)),
            )
        )
    return scored


def _maybe_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class OpenAIWhisperASRService(ASRService):
    """Local Whisper via the reference `openai-whisper` package.

    Kept as an alternative implementation. It shells out to ffmpeg to decode
    audio, so ffmpeg must be on PATH.
    """

    name = "openai-whisper"

    def __init__(self, model_size: str = "base.en", *, device: str | None = None) -> None:
        self.model_size = model_size
        self.device = None if device in (None, "auto") else device
        self._model = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import whisper
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ASRUnavailableError(
                    "openai-whisper is not installed.",
                    detail="pip install openai-whisper (and `brew install ffmpeg`)",
                ) from exc
            logger.info("Loading openai-whisper model %s", self.model_size)
            try:
                self._model = whisper.load_model(self.model_size, device=self.device)
            except Exception as exc:
                raise ASRUnavailableError(
                    f"Could not load Whisper model '{self.model_size}'.",
                    detail=f"{type(exc).__name__}: {exc}",
                ) from exc

    def transcribe(self, audio_path: Path, *, language: str | None = None) -> TranscriptionResult:
        self.load()
        assert self._model is not None
        try:
            with self._infer_lock:
                raw = self._model.transcribe(str(audio_path), language=language, fp16=False)
        except Exception as exc:
            raise TranscriptionError(
                "Whisper failed to transcribe the audio.",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

        segments = [
            TranscriptSegment(
                text=str(s.get("text", "")).strip(),
                start=float(s.get("start", 0.0)),
                end=float(s.get("end", 0.0)),
                avg_logprob=s.get("avg_logprob"),
                no_speech_prob=s.get("no_speech_prob"),
            )
            for s in raw.get("segments", [])
        ]
        return TranscriptionResult(
            text=str(raw.get("text", "")).strip(),
            language=raw.get("language"),
            duration=segments[-1].end if segments else None,
            segments=segments,
        )


class MockASRService(ASRService):
    """Deterministic stand-in. No model download, no inference.

    Set `ASR_BACKEND=mock` to run the API for frontend integration or tests.
    """

    name = "mock"

    def __init__(
        self,
        transcript: str = "could you get me some water",
        *,
        word_probability: float = 0.95,
    ) -> None:
        self.transcript = transcript
        self.word_probability = word_probability
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def transcribe(self, audio_path: Path, *, language: str | None = None) -> TranscriptionResult:
        self.load()
        return TranscriptionResult(
            text=self.transcript,
            language=language or "en",
            duration=1.0,
            segments=[
                TranscriptSegment(
                    text=self.transcript,
                    start=0.0,
                    end=1.0,
                    avg_logprob=-0.25,
                    no_speech_prob=0.01,
                    words=[
                        WordScore(word=word, probability=self.word_probability)
                        for word in self.transcript.split()
                    ],
                )
            ],
        )


def build_asr_service(settings: Settings) -> ASRService:
    """Instantiate the ASR implementation named by settings."""
    if settings.asr_backend == "mock":
        return MockASRService(word_probability=settings.mock_word_probability)
    if settings.asr_backend == "openai-whisper":
        return OpenAIWhisperASRService(settings.whisper_model, device=settings.whisper_device)
    return WhisperASRService(
        settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
