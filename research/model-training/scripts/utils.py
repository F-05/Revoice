"""Shared helpers: text normalization, utterance classification, paths.

Nothing in here is imported from the Revoice application project.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
EVAL_DATA_DIR = PROJECT_ROOT / "evaluation" / "data"
EVAL_REPORT_DIR = PROJECT_ROOT / "evaluation" / "reports"

for _d in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR, EVAL_DATA_DIR, EVAL_REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Text normalization
# --------------------------------------------------------------------------
# Apostrophes and intra-word hyphens are kept so "don't" stays one token and
# "re-do" is not split into two. Everything else that is punctuation is dropped.
_APOSTROPHES = dict.fromkeys(map(ord, "‘’ʼ´`"), "'")
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―"), "-")
_PUNCT_RE = re.compile(r"[^\w\s'\-]", flags=re.UNICODE)
_EDGE_PUNCT_RE = re.compile(r"(?<!\w)['\-]+|['\-]+(?!\w)")
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str | None) -> str:
    """Lowercase, strip punctuation and collapse whitespace, for metric use.

    Used ONLY for scoring; the original strings are always kept alongside.

    >>> normalize_text("  Please, close  the DOOR! ")
    'please close the door'
    >>> normalize_text("Don't -- stop")
    "don't stop"
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.translate(_APOSTROPHES).translate(_DASHES)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)          # drop . , ? ! " ( ) etc.
    text = _EDGE_PUNCT_RE.sub(" ", text)      # drop stray leading/trailing ' and -
    text = _WS_RE.sub(" ", text)
    return text.strip()


def lexical_words(text: str | None) -> list[str]:
    """The normalized tokens used for counting words."""
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def classify_utterance(ground_truth: str | None) -> str:
    """``"word"`` for a single lexical word, ``"sentence"`` otherwise.

    >>> classify_utterance("hill")
    'word'
    >>> classify_utterance("The misguided souls have lost their way.")
    'sentence'
    """
    return "word" if len(lexical_words(ground_truth)) == 1 else "sentence"


def is_scorable(ground_truth: str | None) -> bool:
    """False when the reference normalizes to nothing (jiwer cannot score it)."""
    return bool(lexical_words(ground_truth))
