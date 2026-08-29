"""Public API contract.

Treat everything in this file as a contract with the Expo frontend. Fields may
be ADDED over time, but existing field names, types and nullability should not
change without telling the frontend team.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class UncertainWord(BaseModel):
    """A word the pipeline is not confident about.

    Shaped for the frontend's uncertain screen: it blanks out `position` in
    `raw_transcript` and offers `options` as tappable choices.
    """

    #: Zero-based index of the word inside `raw_transcript.split()`.
    position: int = Field(ge=0)
    #: The word exactly as it appears in `raw_transcript`.
    original: str
    #: Candidate replacements, best first. May be empty -- the frontend always
    #: offers "Something else" and "Say that word again" as well.
    options: list[str] = Field(default_factory=list)


SpeechStatus = Literal["success", "uncertain", "retry"]


class ProcessSpeechResponse(BaseModel):
    """Response for POST /process-speech.

    `status` drives the frontend state machine:

    * ``success``   -- speak `repaired_text` (or `raw_transcript`) straight away
    * ``uncertain`` -- ask the user about `uncertain_words[0]` first
    * ``retry``     -- nothing usable was heard; ask for another recording
    """

    status: SpeechStatus = "success"

    # Verbatim ASR output. Null only when status is "retry".
    raw_transcript: str | None = None

    # Conservative repair of `raw_transcript`. Null when nothing was changed.
    repaired_text: str | None = None

    # Overall confidence in `raw_transcript`, 0.0-1.0. Null when unavailable.
    confidence: float | None = None

    # Words the pipeline is unsure about. Non-empty only when status="uncertain".
    uncertain_words: list[UncertainWord] = Field(default_factory=list)

    # URL of synthesised speech, absolute or relative to the API base
    # (e.g. "/audio/result-ab12.wav"). Null when no audio was produced.
    audio_url: str | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    detail: str | None = None


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    error: ErrorDetail
