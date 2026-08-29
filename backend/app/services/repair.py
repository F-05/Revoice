"""Conservative transcript repair.

The trained repair model is still milestone 2. What ships today is a
deliberately dumb pass-through that only tidies presentation -- capitalisation
and a terminating mark -- so the app has a `repaired_text` to speak and the
route does not have to change when the real model lands.

It never invents, drops or substitutes words.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.config import Settings
from app.services.asr import TranscriptionResult

#: Opening words that make a sentence a question.
_QUESTION_STARTERS = frozenset(
    """could can would will do does did is are am may might should
    who what where when why how""".split()
)


class RepairService(ABC):
    name: str = "repair"

    @abstractmethod
    def repair(self, transcription: TranscriptionResult) -> str | None:
        """Return a conservatively corrected transcript, or None to leave it alone."""


class NoOpRepairService(RepairService):
    """Never rewrite the transcript."""

    name = "noop"

    def repair(self, transcription: TranscriptionResult) -> str | None:
        return None


class PassthroughRepairService(RepairService):
    """Placeholder: the ASR transcript, tidied but not reworded.

    Whisper already punctuates most output, so this is usually a no-op. It
    exists so `repaired_text` is populated end to end before the real model.
    """

    name = "passthrough"

    def repair(self, transcription: TranscriptionResult) -> str | None:
        text = " ".join(transcription.text.split())
        if not text:
            return None

        text = text[0].upper() + text[1:]
        if text[-1] not in ".?!":
            first = text.split(" ", 1)[0].strip(".,?!'\"").lower()
            text += "?" if first in _QUESTION_STARTERS else "."
        return text


def build_repair_service(settings: Settings) -> RepairService:
    """Instantiate the repair implementation. One option for now."""
    return PassthroughRepairService()
