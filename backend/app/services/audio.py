"""Temporary audio file handling and optional preprocessing."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import UploadFile

from app.errors import AudioDecodeError, AudioTooLargeError, InvalidAudioError

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 1024 * 1024  # 1 MiB

# Container formats Expo/expo-audio realistically produces, plus the usual
# suspects for curl-based testing.
SUPPORTED_SUFFIXES = {
    ".wav", ".mp3", ".m4a", ".mp4", ".aac", ".ogg", ".oga", ".opus",
    ".webm", ".flac", ".caf", ".aiff", ".aif", ".3gp", ".amr",
}

TARGET_SAMPLE_RATE = 16_000


@lru_cache
def ffmpeg_path() -> str | None:
    """Absolute path to ffmpeg, or None if it is not installed."""
    return shutil.which("ffmpeg")


def _suffix_for(upload: UploadFile) -> str:
    name = upload.filename or ""
    suffix = Path(name).suffix.lower()
    if suffix and suffix not in SUPPORTED_SUFFIXES:
        raise InvalidAudioError(
            f"Unsupported audio format '{suffix}'.",
            detail="Supported: " + ", ".join(sorted(SUPPORTED_SUFFIXES)),
        )
    # No/unknown extension: let the decoder sniff the container.
    return suffix or ".bin"


async def save_upload_to_temp(
    upload: UploadFile,
    *,
    max_bytes: int,
    temp_dir: str | None = None,
) -> Path:
    """Stream an upload to a temp file, enforcing `max_bytes` as we go.

    Streaming (rather than `await upload.read()`) keeps a hostile or buggy
    client from pinning a large recording in memory.
    """
    suffix = _suffix_for(upload)
    fd, tmp_name = tempfile.mkstemp(prefix="speech-", suffix=suffix, dir=temp_dir)
    path = Path(tmp_name)
    written = 0

    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := await upload.read(_CHUNK_SIZE):
                written += len(chunk)
                if written > max_bytes:
                    raise AudioTooLargeError(
                        f"Audio exceeds the {max_bytes // (1024 * 1024)} MB limit.",
                    )
                out.write(chunk)
    except Exception:
        cleanup(path)
        raise

    if written == 0:
        cleanup(path)
        raise InvalidAudioError("The uploaded audio file is empty.")

    logger.debug("Saved upload %s (%d bytes) to %s", upload.filename, written, path)
    return path


def probe_audio(path: Path) -> float | None:
    """Duration in seconds, or None when the container does not report one.

    Also acts as a gate: a file with no readable audio raises here, with a 422,
    instead of blowing up inside Whisper as an opaque 500. MediaRecorder in a
    browser can produce a header-only clip when a recording is stopped
    immediately, and that should read as "I did not catch that".
    """
    try:
        import av
    except ImportError:  # pragma: no cover - PyAV ships with faster-whisper
        return None

    try:
        with av.open(str(path)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise AudioDecodeError(
                    "The recording contains no audio.",
                    detail="No audio stream in the uploaded container.",
                )

            try:
                next(container.decode(stream))
            except StopIteration:
                raise AudioDecodeError(
                    "The recording contains no audio.",
                    detail="The audio stream is empty.",
                ) from None

            if container.duration:
                return container.duration / av.time_base
            if stream.duration and stream.time_base:
                return float(stream.duration * stream.time_base)
            # Live-recorded webm often has no duration in the header. Unknown
            # is fine -- the ASR backend will decode it in full anyway.
            return None
    except AudioDecodeError:
        raise
    except Exception as exc:
        raise AudioDecodeError(
            "Could not read the uploaded audio.",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


def transcode_to_wav16k(source: Path, *, temp_dir: str | None = None) -> Path:
    """Normalise any container to 16 kHz mono 16-bit WAV using ffmpeg.

    Whisper resamples to 16 kHz mono internally anyway, so this is purely a
    compatibility step for decoders that cannot open exotic containers.
    """
    ffmpeg = ffmpeg_path()
    if ffmpeg is None:
        raise AudioDecodeError("ffmpeg is not installed.")

    fd, tmp_name = tempfile.mkstemp(prefix="speech-", suffix=".wav", dir=temp_dir)
    os.close(fd)
    target = Path(tmp_name)

    cmd = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source),
        "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE), "-f", "wav",
        str(target),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0 or not target.exists() or target.stat().st_size == 0:
        cleanup(target)
        raise AudioDecodeError(
            "Could not decode the uploaded audio.",
            detail=result.stderr.strip()[:500] or None,
        )
    return target


def prepare_audio(source: Path, *, temp_dir: str | None = None, use_ffmpeg: bool = True) -> tuple[Path, list[Path]]:
    """Return the path to feed the ASR service, plus every temp file to clean up.

    When ffmpeg is available we normalise first, because it accepts more
    containers than any single Python decoder. When it is not, we hand the
    original file straight to the ASR backend -- `faster-whisper` decodes
    m4a/webm/ogg itself via PyAV, so ffmpeg is genuinely optional.
    """
    temps = [source]
    if use_ffmpeg and ffmpeg_path() is not None:
        normalised = transcode_to_wav16k(source, temp_dir=temp_dir)
        temps.append(normalised)
        return normalised, temps
    return source, temps


def cleanup(*paths: Path) -> None:
    for path in paths:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:  # pragma: no cover - best effort
            logger.warning("Failed to remove temp file %s", path, exc_info=True)
