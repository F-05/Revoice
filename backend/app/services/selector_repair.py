"""The Revoice constrained hypothesis selector (deployed D3-NL).

The selector NEVER generates text. It chooses exactly one member of the hybrid
hypothesis list (H1 = the production transcript A0, H2..H5 = unique decoder
alternatives) or abstains. Candidate display text is canonicalized (loop
collapse + capitalisation + terminal punctuation) when the list is DEFINED, in
`nbest.build_hybrid_list`; selection then surfaces those exact strings
verbatim — there is no text-changing post-processing after selection. Runtime
invariants re-check that every surfaced string is a member of the list and
that `suggested_text` equals `alternatives[0]`; any violation falls back to A0.

Model artifact: a small JSON (~21-feature scaler + one 16-unit tanh MLP + the
global switching margin). Validation is strict and FAILS CLOSED: any problem
disables repair while transcription continues untouched.

Research provenance: the citable research result is the frozen D2 LOSO
evaluation (WER 0.3175 -> 0.2849). The deployed weights are the D3-NL
development-stage selector, chosen for its stronger safety profile. It is not
independently validated on unseen speakers.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.config import Settings
from app.services.asr import TranscriptionResult
from app.services.nbest import Candidate, build_hybrid_list, normalize_text
from app.services.repair import PassthroughRepairService

logger = logging.getLogger(__name__)

EXPECTED_MODEL_VERSION = "revoice_selector_v1"
EXPECTED_FEATURE_VERSION = "d3-21f-v1"
EXPECTED_FEATURES = [
    "is_a0", "rank", "a0_conf", "a0_conf_missing",
    "ct2_score", "ct2_score_rel_best", "ct2_score_gap_next",
    "ct2_score_per_word", "ct2_score_missing",
    "edit_to_a0", "edit_to_a0_norm", "n_words", "len_ratio_median",
    "consensus_f1", "support_frac_ge2", "disputed_support",
    "changed_support_count", "max_ally_overlap", "novel_word_flag",
    "deletion_only_flag", "score_within_list",
]


# --------------------------------------------------------------------------
# text helpers
# --------------------------------------------------------------------------
def _word_edit(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[n]


def _overlap_f1(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def extract_features(candidates: list[Candidate], a0_conf: float | None) -> np.ndarray:
    """The frozen 21 features, in EXPECTED_FEATURES order, one row per candidate."""
    texts = [c.normalized for c in candidates]
    a0 = next((c.normalized for c in candidates if c.is_a0), "")
    a0_words, a0_set = a0.split(), set(a0.split())
    word_sets = [set(t.split()) for t in texts]
    global_support = Counter(w for ws in word_sets for w in ws)
    ct2 = [c.ct2_score for c in candidates if c.ct2_score is not None]
    best_ct2 = max(ct2) if ct2 else 0.0
    sorted_ct2 = sorted(ct2, reverse=True)
    lo, hi = (min(ct2), max(ct2)) if ct2 else (0.0, 0.0)
    spread = hi - lo
    lengths = [len(t.split()) for t in texts]
    median_len = float(np.median([l for l in lengths if l > 0]) or 1.0)

    rows = []
    for i, c in enumerate(candidates):
        words = texts[i].split()
        wset = word_sets[i]
        others = [word_sets[j] for j in range(len(texts)) if j != i]
        cons = float(np.mean([_overlap_f1(wset, o) for o in others])) if others else 1.0
        sup2 = float(np.mean([global_support[w] >= 3 for w in words])) if words else 0.0
        changed = [w for w in words if w not in a0_set]
        deleted = [w for w in a0_set if w not in wset]
        dsup = float(np.mean([global_support[w] >= 2 for w in changed])) if changed else 1.0
        ed = _word_edit(words, a0_words)
        score = c.ct2_score
        gap = 0.0
        if score is not None and len(sorted_ct2) > 1:
            pos = sorted_ct2.index(score)
            if pos + 1 < len(sorted_ct2):
                gap = sorted_ct2[pos] - sorted_ct2[pos + 1]
        if changed:
            support_count = float(np.median(
                [sum(w in o for o in others) for w in changed]))
        else:
            support_count = float(len(others)) if c.is_a0 else 0.0
        ally = float(max([_overlap_f1(wset, o) for o in others] or [1.0]))
        rows.append([
            float(c.is_a0), float(i),
            float(a0_conf) if (c.is_a0 and a0_conf is not None) else 0.0,
            float(c.is_a0 and a0_conf is None),
            float(score) if score is not None else 0.0,
            float(score - best_ct2) if score is not None else 0.0,
            float(gap),
            float(score) / max(len(words), 1) if score is not None else 0.0,
            float(score is None),
            float(ed), float(ed) / max(len(words), len(a0_words), 1),
            float(len(words)), float(len(words)) / median_len if median_len else 0.0,
            cons, sup2, dsup,
            support_count, ally,
            float(any(global_support[w] == 1 for w in changed)),
            float((not c.is_a0) and (not changed) and bool(deleted)),
            float((score - lo) / spread) if score is not None and spread > 0 else 0.0,
        ])
    return np.array(rows, dtype=np.float64)


# --------------------------------------------------------------------------
# artifact
# --------------------------------------------------------------------------
class SelectorModel:
    """Frozen NumPy-only inference. Raises ValueError on any validation issue."""

    def __init__(self, artifact: dict):
        if artifact.get("model_version") != EXPECTED_MODEL_VERSION:
            raise ValueError(f"unsupported model_version {artifact.get('model_version')!r}")
        if artifact.get("feature_version") != EXPECTED_FEATURE_VERSION:
            raise ValueError(f"unsupported feature_version {artifact.get('feature_version')!r}")
        if artifact.get("features") != EXPECTED_FEATURES:
            raise ValueError("feature list/order does not match this build")
        self.mu = np.asarray(artifact["scaler_mean"], dtype=np.float64)
        self.sd = np.asarray(artifact["scaler_std"], dtype=np.float64)
        self.W1 = np.asarray(artifact["W1"], dtype=np.float64)
        self.b1 = np.asarray(artifact["b1"], dtype=np.float64)
        self.W2 = np.asarray(artifact["W2"], dtype=np.float64)
        self.b2 = float(artifact["b2"])
        n = len(EXPECTED_FEATURES)
        if not (self.mu.shape == (n,) and self.sd.shape == (n,)
                and self.W1.shape[1] == n and self.W1.shape[0] == self.b1.shape[0]
                and self.W2.shape == (self.W1.shape[0],)):
            raise ValueError("parameter dimensions do not match the 21-feature model")
        policy = artifact.get("policy") or {}
        if policy.get("auto_switch_enabled", False):
            raise ValueError("auto_switch_enabled=true is not supported by this build")
        self.suggestion_tau = float(policy.get("suggestion_tau", 0.35))
        if not (0.0 <= self.suggestion_tau <= 1.0):
            raise ValueError(f"suggestion_tau {self.suggestion_tau} outside [0, 1]")
        self.whisper_model = artifact.get("whisper_model")

    def scores(self, feats: np.ndarray) -> np.ndarray:
        x = (feats - self.mu) / self.sd
        return np.tanh(x @ self.W1.T + self.b1) @ self.W2 + self.b2

    def probabilities(self, feats: np.ndarray) -> np.ndarray:
        s = self.scores(feats)
        e = np.exp(s - s.max())
        return e / e.sum()


@dataclass(slots=True)
class SelectorOutcome:
    """Suggestion-first result. `SWITCH` is never emitted in new-user mode."""

    decision: str                      # KEEP_A0 | UNCERTAIN
    final_display: str                 # ALWAYS A0 (tidied) in this policy
    final_normalized: str
    #: The selector's preferred non-A0 hypothesis (tidied), when it prefers one.
    suggested_text: str | None = None
    #: Internal/debug only: "strong" when margin >= suggestion_tau.
    suggestion_strength: str | None = None
    alternatives: list[str] = field(default_factory=list)  # recommended first
    margin: float | None = None
    n_candidates: int = 0
    nbest_ms: float | None = None
    selector_ms: float | None = None


class SelectorRepairService(PassthroughRepairService):
    """Repair via constrained hypothesis selection.

    Inherits the passthrough `repair()` so anything that cannot reach the
    selector (no audio path, no real Whisper model, load failure) behaves
    exactly like the previous backend.
    """

    name = "selector"

    def __init__(self, settings: Settings):
        self._model: SelectorModel | None = None
        self._disabled_reason: str | None = None
        path = Path(settings.repair_model_path or "")
        try:
            if not path.is_file():
                raise FileNotFoundError(f"repair model artifact not found: {path}")
            artifact = json.loads(path.read_text())
            self._model = SelectorModel(artifact)
            if settings.repair_switch_margin is not None:
                logger.warning(
                    "REPAIR_SWITCH_MARGIN override active: %.3f (artifact "
                    "suggestion_tau %.3f). This controls SUGGESTION STRENGTH "
                    "only; automatic switching stays disabled.",
                    settings.repair_switch_margin, self._model.suggestion_tau)
                self._model.suggestion_tau = float(settings.repair_switch_margin)
            if (self._model.whisper_model and settings.whisper_model
                    != self._model.whisper_model):
                logger.warning(
                    "WHISPER_MODEL=%s but the selector was calibrated on %s — "
                    "quoted repair metrics do not apply to this configuration",
                    settings.whisper_model, self._model.whisper_model)
            logger.info(
                "selector loaded: suggestion-first, auto_switch=disabled, "
                "suggestion_tau=%.3f, 21 features, %s",
                self._model.suggestion_tau, path.name)
        except Exception as exc:  # noqa: BLE001 — FAIL CLOSED to A0-only
            self._disabled_reason = f"{type(exc).__name__}: {exc}"
            logger.error("selector disabled, falling back to A0-only: %s",
                         self._disabled_reason)

    @property
    def active(self) -> bool:
        return self._model is not None

    def select(self, audio_path: Path, transcription: TranscriptionResult,
               asr_service) -> SelectorOutcome | None:
        """Run the full constrained selection. None -> caller uses legacy path."""
        if self._model is None:
            return None
        whisper = getattr(asr_service, "_model", None)
        if whisper is None:
            logger.debug("selector: no loaded whisper model (mock backend?); skipping")
            return None
        a0_text = transcription.text.strip()
        if not a0_text:
            return None
        try:
            t0 = time.perf_counter()
            candidates = build_hybrid_list(whisper, audio_path, a0_text)
            nbest_ms = (time.perf_counter() - t0) * 1000
            if not candidates:
                return None
            t1 = time.perf_counter()
            outcome = self.decide(candidates, transcription.confidence)
            outcome.nbest_ms = nbest_ms
            outcome.selector_ms = (time.perf_counter() - t1) * 1000
            return outcome
        except Exception:  # noqa: BLE001 — never break transcription
            logger.exception("selector failed; returning A0")
            return None

    def decide(self, candidates: list[Candidate],
               a0_conf: float | None) -> SelectorOutcome:
        """Pure decision logic (unit-testable without audio/Whisper)."""
        assert self._model is not None
        a0_idx = next((i for i, c in enumerate(candidates) if c.is_a0), None)
        texts_norm = [c.normalized for c in candidates]

        suggested_idx: int | None = None
        margin: float | None = None
        strength: str | None = None
        if a0_idx is None:
            # No usable A0 in the list (empty transcript) — nothing to keep or
            # suggest against; behave as legacy.
            a0_idx = 0
            decision = "KEEP_A0"
        elif len(candidates) == 1:
            decision = "KEEP_A0"
        else:
            probs = self._model.probabilities(
                extract_features(candidates, a0_conf))
            p_a0 = float(probs[a0_idx])
            alt = [(float(probs[i]), i) for i in range(len(candidates)) if i != a0_idx]
            p_alt, alt_idx = max(alt)
            margin = p_alt - p_a0
            if p_alt > p_a0:
                # Suggestion-first policy: NEVER auto-switch for a new user.
                decision = "UNCERTAIN"
                suggested_idx = alt_idx
                strength = ("strong" if margin >= self._model.suggestion_tau - 1e-12
                            else "weak")
            else:
                decision = "KEEP_A0"

        # Output is ALWAYS A0 in this policy; suggestion rides alongside.
        chosen = candidates[a0_idx]
        suggested = candidates[suggested_idx] if suggested_idx is not None else None
        # HARD INVARIANTS: everything surfaced must be a member of the list.
        if chosen.normalized not in texts_norm or (
                suggested is not None and suggested.normalized not in texts_norm
        ):  # pragma: no cover — must be impossible
            logger.error("SELECTOR INVARIANT VIOLATED; returning bare A0")
            return SelectorOutcome(decision="KEEP_A0",
                                   final_display=chosen.display,
                                   final_normalized=chosen.normalized,
                                   n_candidates=len(candidates))
        # Recommended candidate first, then the remaining non-A0 hypotheses.
        # Strings are surfaced VERBATIM from the candidate list — display text
        # was canonicalized when the list was defined, and nothing rewrites it
        # after selection.
        rest = [c.display for c in candidates
                if not c.is_a0 and c is not suggested]
        ordered = ([suggested.display] if suggested is not None else []) + rest
        alternatives = ordered[:3]
        suggested_text = suggested.display if suggested is not None else None
        # Runtime invariant: the suggestion is exactly alternatives[0].
        if suggested_text is not None and (
                not alternatives or alternatives[0] != suggested_text
        ):  # pragma: no cover — must be impossible
            logger.error("SUGGESTION INVARIANT VIOLATED; returning bare A0")
            return SelectorOutcome(decision="KEEP_A0",
                                   final_display=chosen.display,
                                   final_normalized=chosen.normalized,
                                   n_candidates=len(candidates))
        return SelectorOutcome(
            decision=decision,
            final_display=chosen.display,
            final_normalized=chosen.normalized,
            suggested_text=suggested_text,
            suggestion_strength=strength,
            alternatives=alternatives,
            margin=margin, n_candidates=len(candidates))


