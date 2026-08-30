"""Shared error-triage heuristics (severity + estimated repairability).

Extracted verbatim from the first (770-sample) experiment so that the small
base.en baseline and the larger TORGO experiment are scored by an identical
rule and stay comparable. Do not change these thresholds without re-running
BOTH experiments.

These are deterministic analysis heuristics, not clinically or scientifically
validated categories. `repairability` estimates whether enough context survives
for a text-only repair model to have a chance; it is not evidence that any
sentence can actually be repaired.
"""

from __future__ import annotations

import jiwer


# --------------------------------------------------------------------------
# Per-utterance edit statistics
# --------------------------------------------------------------------------
def edit_stats(reference: str, hypothesis: str) -> dict:
    """Word-level hits/substitutions/deletions/insertions plus derived ratios."""
    out = jiwer.process_words([reference], [hypothesis if hypothesis.strip() else "*"])
    ref_len = len(reference.split())
    hyp_len = len(hypothesis.split())
    total_errors = out.substitutions + out.deletions + out.insertions
    return {
        "wer": out.wer,
        "hits": out.hits,
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "total_errors": total_errors,
        "ref_words": ref_len,
        "hyp_words": hyp_len,
        # Share of the reference Whisper got right and in the right place.
        "content_preserved": out.hits / ref_len if ref_len else 0.0,
        # >1 means Whisper produced more words than were spoken.
        "length_ratio": hyp_len / ref_len if ref_len else 0.0,
    }


# --------------------------------------------------------------------------
# Heuristic 1 -- severity
# --------------------------------------------------------------------------
SEVERITY_RULE = """\
Applied in order, on normalized text, to sentences only (word-level edits via
jiwer). `preserved` = hits / reference_words.

  MINOR     wer <= 0.34  AND  total_errors <= 2  AND  preserved >= 0.66
            -> most of the sentence survives, roughly one local lexical slip.
  SEVERE    wer >= 0.75  OR   preserved <= 0.34  OR  hypothesis is empty
            -> most of the intended sentence is missing, hallucinated or
               unrelated.
  MODERATE  everything else
            -> several errors, but a substantial part of the sentence remains.
"""


def severity(stats: dict, hypothesis: str) -> str:
    if not hypothesis.strip():
        return "SEVERE"
    if stats["wer"] <= 0.34 and stats["total_errors"] <= 2 and stats["content_preserved"] >= 0.66:
        return "MINOR"
    if stats["wer"] >= 0.75 or stats["content_preserved"] <= 0.34:
        return "SEVERE"
    return "MODERATE"


# --------------------------------------------------------------------------
# Heuristic 2 -- estimated repairability
# --------------------------------------------------------------------------
REPAIRABILITY_RULE = """\
Deliberately conservative, applied in order. This estimates whether ENOUGH
CONTEXT SURVIVES for a text-only repair model to have a chance -- it is not a
claim that any sentence can actually be repaired.

  LOW     hypothesis empty
          OR preserved <= 0.34            (little of the sentence survives)
          OR wer >= 0.75
          OR length_ratio > 1.75          (long hallucination bolted on)
          OR length_ratio < 0.50          (most of the sentence dropped)
  HIGH    preserved >= 0.70
          AND total_errors <= 2
          AND deletions + insertions <= 1 (errors are substitutions, i.e. local)
          AND ref_words >= 4              (enough context to condition on)
          AND 0.80 <= length_ratio <= 1.25
  MEDIUM  everything else
"""


def repairability(stats: dict, hypothesis: str) -> str:
    if (not hypothesis.strip()
            or stats["content_preserved"] <= 0.34
            or stats["wer"] >= 0.75
            or stats["length_ratio"] > 1.75
            or stats["length_ratio"] < 0.50):
        return "LOW"
    if (stats["content_preserved"] >= 0.70
            and stats["total_errors"] <= 2
            and stats["deletions"] + stats["insertions"] <= 1
            and stats["ref_words"] >= 4
            and 0.80 <= stats["length_ratio"] <= 1.25):
        return "HIGH"
    return "MEDIUM"


def edit_profile(stats: dict, hypothesis: str) -> str:
    """Which edit type dominates -- useful when reading the error CSV."""
    if not hypothesis.strip():
        return "empty_output"
    s, d, i = stats["substitutions"], stats["deletions"], stats["insertions"]
    if s and not d and not i:
        return "substitution_only"
    if i > s and i >= d:
        return "insertion_heavy"
    if d > s and d >= i:
        return "deletion_heavy"
    return "mixed"


def repair_reason(sev: str, rep: str, stats: dict, hypothesis: str) -> str:
    if not hypothesis.strip():
        return "Whisper produced no output at all, so there is no text to repair."
    if rep == "HIGH":
        return (f"{stats['content_preserved']:.0%} of the reference words survive and the "
                f"error is {stats['total_errors']} local substitution(s); a text model may be "
                "able to infer the intended word from the surrounding context.")
    if rep == "LOW":
        if stats["length_ratio"] > 1.75:
            return ("Whisper produced far more words than were spoken, so the output is "
                    "largely hallucinated and the intended sentence is not recoverable from text.")
        if stats["length_ratio"] < 0.50:
            return ("Most of the sentence is missing from the output; too little survives for "
                    "a text-only model to reconstruct it.")
        return (f"Only {stats['content_preserved']:.0%} of the reference survives "
                f"(WER {stats['wer']:.2f}); too much information is lost for text-only repair.")
    return (f"{stats['content_preserved']:.0%} of the reference survives with "
            f"{stats['total_errors']} errors; some context remains but several words are "
            "ambiguous, so repair is uncertain.")
