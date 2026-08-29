"""Text-to-speech.

The clarified sentence is spoken by the backend when a system voice is
available, and the resulting file is served from `/audio`. The frontend falls
back to the device voice whenever `audio_url` is null, so this is an
enhancement rather than a hard dependency.

Nothing here is the final voice -- milestone 4 replaces `SystemTTSService`.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from app.config import Settings

logger = logging.getLogger(__name__)

#: URL prefix the generated files are served from. Must match the mount in main.py.
AUDIO_URL_PREFIX = "/audio"

_SAMPLE_RATE = 22_050


class TTSService(ABC):
    name: str = "tts"

    @abstractmethod
    def synthesize(self, text: str) -> str | None:
        """Synthesize `text` and return a URL path the client can fetch, or None."""


class NoOpTTSService(TTSService):
    """Produce no audio. The app speaks with the device voice instead."""

    name = "noop"

    def synthesize(self, text: str) -> str | None:
        return None


class SystemTTSService(TTSService):
    """macOS `say`, written straight to a 16-bit mono WAV.

    Chosen because it needs no model download, no network and no extra
    dependency -- enough to prove the audio leg of the pipeline.
    """

    name = "system"

    def __init__(self, output_dir: Path, *, keep_files: int = 50) -> None:
        self.output_dir = output_dir
        self.keep_files = keep_files
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def is_available() -> bool:
        return shutil.which("say") is not None

    def synthesize(self, text: str) -> str | None:
        text = text.strip()
        if not text:
            return None

        say = shutil.which("say")
        if say is None:
            logger.warning("System TTS requested but `say` is not on PATH.")
            return None

        target = self.output_dir / f"result-{uuid.uuid4().hex[:12]}.wav"
        cmd = [
            say, "-o", str(target),
            "--data-format=LEI16@%d" % _SAMPLE_RATE,
            "--channels=1",
            "--", text,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("System TTS failed: %s: %s", type(exc).__name__, exc)
            return None

        if result.returncode != 0 or not target.exists() or target.stat().st_size == 0:
            logger.warning("System TTS produced no audio: %s", result.stderr.strip()[:300])
            target.unlink(missing_ok=True)
            return None

        self._prune()
        # Relative on purpose: the frontend resolves it against its API base URL,
        # which differs per device (localhost, LAN IP, tunnel).
        return f"{AUDIO_URL_PREFIX}/{target.name}"

    def _prune(self) -> None:
        """Keep the directory from growing without bound across a long session."""
        try:
            files = sorted(
                self.output_dir.glob("result-*.wav"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            for stale in files[self.keep_files:]:
                stale.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - best effort
            logger.debug("Could not prune %s", self.output_dir, exc_info=True)


def resolve_audio_dir(settings: Settings) -> Path:
    """Directory generated clips are written to and served from."""
    if settings.audio_output_dir:
        return Path(settings.audio_output_dir)
    import tempfile

    return Path(tempfile.gettempdir()) / "speech-repair-audio"


def build_tts_service(settings: Settings) -> TTSService:
    """Instantiate the TTS implementation named by settings.

    Falls back to no-op (rather than failing the request) when the chosen
    backend is not usable on this machine.
    """
    if settings.tts_backend == "system":
        if not SystemTTSService.is_available():
            logger.warning("TTS_BACKEND=system but `say` is unavailable; using no-op TTS.")
            return NoOpTTSService()
        return SystemTTSService(
            resolve_audio_dir(settings), keep_files=settings.audio_keep_files
        )
    return NoOpTTSService()
