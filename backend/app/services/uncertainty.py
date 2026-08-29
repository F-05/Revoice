"""Turning raw ASR scores into the response status the frontend switches on.

Kept out of the route so the thresholds and the word-picking rule live in one
place. This is intentionally simple; the calibrated version arrives with
milestone 3.
"""

from __future__ import annotations

from app.config import Settings
from app.models.schemas import SpeechStatus, UncertainWord
from app.services.asr import TranscriptionResult

#: Stripped from a word before it is offered back to the user as an option.
_EDGE_PUNCTUATION = " .,?!;:\"'"


def find_uncertain_word(
    transcription: TranscriptionResult, settings: Settings
) -> UncertainWord | None:
    """The least confident word, if it falls below the threshold.

    One word at a time on purpose: the frontend asks about `uncertain_words[0]`
    and a single question is far easier to answer than a list of them.
    """
    words = transcription.words
    if not words:
        return None

    tokens = transcription.text.split()
    index, weakest = min(enumerate(words), key=lambda pair: pair[1].probability)
    if weakest.probability >= settings.uncertain_word_threshold:
        return None

    # `position` indexes into `raw_transcript.split()`, which is what the
    # frontend splits on. The engine's word list normally lines up 1:1; when it
    # does not, fall back to locating the word by text.
    if index >= len(tokens) or tokens[index].strip(_EDGE_PUNCTUATION) != weakest.word.strip(
        _EDGE_PUNCTUATION
    ):
        target = weakest.word.strip(_EDGE_PUNCTUATION).lower()
        index = next(
            (i for i, token in enumerate(tokens) if token.strip(_EDGE_PUNCTUATION).lower() == target),
            -1,
        )
        if index < 0:
            return None

    heard = tokens[index].strip(_EDGE_PUNCTUATION)
    return UncertainWord(
        position=index,
        original=tokens[index],
        # No alternatives generator yet -- offer what was heard so the user can
        # confirm it. The frontend always adds "Something else" itself.
        options=[heard] if heard else [],
    )


def decide_status(
    transcription: TranscriptionResult,
    uncertain_word: UncertainWord | None,
    settings: Settings,
) -> SpeechStatus:
    """Pick `success` / `uncertain` / `retry` for a transcription."""
    if not transcription.text.strip():
        return "retry"

    confidence = transcription.confidence
    if confidence is not None and confidence < settings.retry_confidence_threshold:
        return "retry"

    return "uncertain" if uncertain_word is not None else "success"
