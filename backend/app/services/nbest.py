"""N-best hypothesis generation for the Revoice selector.

Reuses the already-loaded faster-whisper model to run one extra CTranslate2
beam-search pass and returns the hybrid candidate list:

    H1 = the production transcript (A0), exactly as `ASRService` produced it
    H2..H5 = up to four unique ct2 alternatives

The decode configuration is FROZEN — it matches the research runs byte for
byte (beam 12, 8 hypotheses, Silero VAD trim with production defaults,
reference-independent loop-collapse, dedupe on normalized text). Do not tune
it here; retraining the selector is required if it changes.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

BEAM_SIZE = 12
NUM_HYPOTHESES = 8
KEEP_UNIQUE = 5  # including A0

_QUESTION_STARTERS = frozenset(
    """could can would will do does did is are am may might should
    who what where when why how""".split()
)


def canonical_display(text: str) -> str:
    """Presentational canonicalization applied when the candidate list is
    DEFINED — capitalise + terminal mark, words never change. Selection and
    the API surface these exact strings; nothing rewrites them afterwards."""
    text = " ".join(text.split())
    if not text:
        return text
    text = text[0].upper() + text[1:]
    if text[-1] not in ".?!":
        first = text.split(" ", 1)[0].strip(".,?!'\"").lower()
        text += "?" if first in _QUESTION_STARTERS else "."
    return text

# --- normalization (identical to the research `normalize_text`) -----------
_APOSTROPHES = dict.fromkeys(map(ord, "‘’ʼ´`"), "'")
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―"), "-")
_PUNCT_RE = re.compile(r"[^\w\s'\-]", flags=re.UNICODE)
_EDGE_PUNCT_RE = re.compile(r"(?<!\w)['\-]+|['\-]+(?!\w)")
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.translate(_APOSTROPHES).translate(_DASHES)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _EDGE_PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def collapse_loops(tokens: list[str]) -> list[str]:
    """Collapse a phrase of >=2 words repeated consecutively >=2 times."""
    changed = True
    while changed:
        changed = False
        for period in range(min(len(tokens) // 2, 12), 1, -1):
            i = 0
            while i + 2 * period <= len(tokens):
                if tokens[i:i + period] == tokens[i + period:i + 2 * period]:
                    del tokens[i + period:i + 2 * period]
                    changed = True
                else:
                    i += 1
    return tokens


@dataclass(slots=True)
class Candidate:
    display: str          # what the user may eventually see (raw decoder text)
    normalized: str       # what the selector scores
    is_a0: bool
    ct2_score: float | None


def build_hybrid_list(model, audio_path: Path, a0_text: str) -> list[Candidate] | None:
    """H1=A0 plus up to 4 unique ct2 alternatives. None when generation fails."""
    try:
        import ctranslate2
        from faster_whisper.audio import decode_audio
        from faster_whisper.vad import VadOptions, collect_chunks, get_speech_timestamps

        started = time.perf_counter()
        audio = decode_audio(str(audio_path), sampling_rate=16_000)
        chunks = get_speech_timestamps(audio, VadOptions())
        if chunks:
            pieces = collect_chunks(audio, chunks)
            audio = np.concatenate(pieces) if isinstance(pieces, list) else pieces
        if len(audio) < 160:
            audio = decode_audio(str(audio_path), sampling_rate=16_000)

        features = model.feature_extractor(audio, padding=True)[:, :3000]
        fv = ctranslate2.StorageView.from_array(
            np.ascontiguousarray(features[None]).astype(np.float32))
        prompt = [model.hf_tokenizer.token_to_id(t)
                  for t in ["<|startoftranscript|>", "<|notimestamps|>"]]
        result = model.model.generate(
            fv, [prompt], beam_size=BEAM_SIZE, num_hypotheses=NUM_HYPOTHESES,
            return_scores=True, max_length=200)[0]

        a0_norm = normalize_text(a0_text)
        candidates: list[Candidate] = []
        seen: set[str] = set()
        if a0_norm:
            candidates.append(Candidate(canonical_display(a0_text), a0_norm, True, None))
            seen.add(a0_norm)
        for seq, score in zip(result.sequences_ids, result.scores):
            raw = model.hf_tokenizer.decode(seq).strip()
            norm = " ".join(collapse_loops(normalize_text(raw).split()))
            # Loop-collapse + tidy happen HERE, before the H1-H5 list exists.
            # The resulting strings are the exact, final candidate texts.
            display = canonical_display(" ".join(collapse_loops(raw.split())))
            if norm and norm not in seen:
                seen.add(norm)
                candidates.append(Candidate(display, norm, False, float(score)))
            if len(candidates) >= KEEP_UNIQUE:
                break
        logger.debug("n-best: %d candidates in %.0f ms",
                     len(candidates), (time.perf_counter() - started) * 1000)
        return candidates or None
    except Exception:  # noqa: BLE001 — repair failure must never break transcription
        logger.exception("N-best generation failed; selector will fall back to A0")
        return None
