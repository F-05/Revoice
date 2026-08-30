"""Steps 7-8 -- compare ASR models on one fixed subset, then pick one.

    python scripts/lt_asr_compare.py

Every model sees the SAME audio with the SAME decoding settings; only the model
name changes.

Selection runs on the VALIDATION speakers. That is what validation is for:
train generates repair-model training data, validation selects the ASR model
and the T5 checkpoint, and test is not read until the final evaluation.

Writes results/large_torgo/asr_model_comparison.csv and
results/large_torgo/asr_selection.json.
"""

from __future__ import annotations

import json
import math
import resource
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_triage import edit_stats, repairability, severity  # noqa: E402
from lt_audio import iter_audio  # noqa: E402
from utils import normalize_text  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
LT_DATA = PROJECT / "data" / "large_torgo"
LT_RESULTS = PROJECT / "results" / "large_torgo"

CANDIDATES = ["small.en", "medium.en", "large-v3-turbo"]

# Held fixed across every candidate. Only `model_size` varies.
DECODE = {
    "language": "en",
    "beam_size": 5,
    "condition_on_previous_text": False,
    "vad_filter": True,          # same setting as the base.en baseline
    "word_timestamps": True,
}
DEVICE = "auto"
COMPUTE_TYPE = "int8"

SELECTION_SPLIT = "validation"


def build_subset() -> pd.DataFrame:
    """Every validation-speaker sentence. Small enough to use whole."""
    splits = pd.read_csv(LT_DATA / "splits.csv", keep_default_na=False, na_values=[""])
    subset = splits[splits["split"] == SELECTION_SPLIT]
    return subset.sort_values("sample_id").reset_index(drop=True)


def transcribe_all(model_size: str, subset: pd.DataFrame, tmp_dir: Path) -> dict:
    from faster_whisper import WhisperModel

    peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    load_started = time.perf_counter()
    model = WhisperModel(model_size, device=DEVICE, compute_type=COMPUTE_TYPE)
    load_seconds = time.perf_counter() - load_started

    rows, audio_seconds, decode_seconds = [], 0.0, 0.0
    ids = list(subset["sample_id"])
    truth = dict(zip(subset["sample_id"], subset["ground_truth"]))

    for sample_id, raw in tqdm(iter_audio(ids), total=len(ids), unit="clip",
                               desc=model_size, dynamic_ncols=True):
        clip = tmp_dir / "clip.wav"
        clip.write_bytes(raw)
        started = time.perf_counter()
        segments_iter, info = model.transcribe(str(clip), **DECODE)
        segments = list(segments_iter)
        decode_seconds += time.perf_counter() - started
        audio_seconds += float(getattr(info, "duration", 0.0) or 0.0)

        text = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
        scored = [s for s in segments if getattr(s, "avg_logprob", None) is not None]
        confidence = None
        if scored:
            weights = [max(s.end - s.start, 1e-3) for s in scored]
            total = sum(weights)
            if total > 0:
                confidence = min(max(sum(math.exp(s.avg_logprob) * w
                                         for s, w in zip(scored, weights)) / total, 0.0), 1.0)
        rows.append({
            "sample_id": sample_id,
            "ground_truth": truth[sample_id],
            "asr_transcript": text,
            "asr_confidence": confidence,
        })

    peak_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    del model

    df = pd.DataFrame(rows)
    df["ground_truth_normalized"] = df["ground_truth"].map(normalize_text)
    df["asr_transcript_normalized"] = df["asr_transcript"].map(normalize_text)

    stats = [edit_stats(r, h) for r, h in
             zip(df["ground_truth_normalized"], df["asr_transcript_normalized"])]
    df["wer"] = [s["wer"] for s in stats]
    df["severity"] = [severity(s, h) if r != h else "CORRECT" for s, h, r in
                      zip(stats, df["asr_transcript_normalized"], df["ground_truth_normalized"])]
    df["repairability"] = [repairability(s, h) if r != h else "CORRECT" for s, h, r in
                           zip(stats, df["asr_transcript_normalized"],
                               df["ground_truth_normalized"])]
    df["model"] = model_size

    import jiwer
    out = jiwer.process_words(
        list(df["ground_truth_normalized"]),
        [h if h.strip() else "*" for h in df["asr_transcript_normalized"]])
    exact = int((df["ground_truth_normalized"] == df["asr_transcript_normalized"]).sum())
    n = len(df)
    incorrect = df[df["severity"] != "CORRECT"]

    metrics = {
        "model": model_size,
        "samples": n,
        "wer": out.wer,
        "exact_match_accuracy": exact / n,
        "num_exact_match": exact,
        "severe_count": int((incorrect["severity"] == "SEVERE").sum()),
        "severe_rate_of_all": float((incorrect["severity"] == "SEVERE").sum()) / n,
        "moderate_count": int((incorrect["severity"] == "MODERATE").sum()),
        "minor_count": int((incorrect["severity"] == "MINOR").sum()),
        "high_repairability": int((incorrect["repairability"] == "HIGH").sum()),
        "medium_repairability": int((incorrect["repairability"] == "MEDIUM").sum()),
        "low_repairability": int((incorrect["repairability"] == "LOW").sum()),
        "load_seconds": round(load_seconds, 2),
        "decode_seconds": round(decode_seconds, 2),
        "seconds_per_clip": round(decode_seconds / n, 3),
        "realtime_factor": round(decode_seconds / audio_seconds, 3) if audio_seconds else None,
        "peak_rss_delta_mb": round((peak_after - peak_before) / 1e6, 1),
        "peak_rss_mb": round(peak_after / 1e6, 1),
    }
    return {"metrics": metrics, "predictions": df}


def select(metrics: list[dict]) -> tuple[str, str]:
    """Pick a model: severe-error rate first, then WER, then exact match.

    A larger model only wins if it is meaningfully better, not merely bigger:
    a rival must beat the incumbent's severe rate by more than 2 percentage
    points (or match it and beat WER by more than 0.02) to displace it. Ties go
    to the cheaper model, which is the one earlier in CANDIDATES.
    """
    ordered = sorted(metrics, key=lambda m: CANDIDATES.index(m["model"]))
    best = ordered[0]
    reasons = []
    for rival in ordered[1:]:
        severe_gain = best["severe_rate_of_all"] - rival["severe_rate_of_all"]
        wer_gain = best["wer"] - rival["wer"]
        if severe_gain > 0.02 or (abs(severe_gain) <= 0.02 and wer_gain > 0.02):
            reasons.append(
                f"{rival['model']} beats {best['model']}: severe rate "
                f"{rival['severe_rate_of_all']:.3f} vs {best['severe_rate_of_all']:.3f}, "
                f"WER {rival['wer']:.4f} vs {best['wer']:.4f}")
            best = rival
        else:
            reasons.append(
                f"{rival['model']} does not clear the margin over {best['model']} "
                f"(severe {rival['severe_rate_of_all']:.3f} vs {best['severe_rate_of_all']:.3f}, "
                f"WER {rival['wer']:.4f} vs {best['wer']:.4f}); keeping the cheaper model")
    return best["model"], " | ".join(reasons)


def main() -> None:
    subset = build_subset()
    print(f"Comparison subset: {len(subset)} dysarthric sentences from "
          f"{subset['speaker_id'].nunique()} VALIDATION speakers "
          f"({dict(subset['speaker_id'].value_counts())}). "
          "Test speakers are not touched.")
    print(f"Fixed decoding: {DECODE}, device={DEVICE}, compute_type={COMPUTE_TYPE}\n")

    all_metrics, frames = [], []
    with tempfile.TemporaryDirectory(prefix="lt-asr-") as tmp:
        for model_size in CANDIDATES:
            print(f"\n--- {model_size} ---", flush=True)
            result = transcribe_all(model_size, subset, Path(tmp))
            all_metrics.append(result["metrics"])
            frames.append(result["predictions"])
            m = result["metrics"]
            print(f"  WER={m['wer']:.4f} exact={m['exact_match_accuracy']:.3f} "
                  f"severe={m['severe_count']}/{m['samples']} "
                  f"({m['severe_rate_of_all']:.1%}) "
                  f"{m['seconds_per_clip']}s/clip rtf={m['realtime_factor']}")

    comparison = pd.DataFrame(all_metrics)
    comparison.to_csv(LT_RESULTS / "asr_model_comparison.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(
        LT_RESULTS / "asr_comparison_predictions.csv", index=False)

    chosen, reasoning = select(all_metrics)
    (LT_RESULTS / "asr_selection.json").write_text(json.dumps({
        "candidates": CANDIDATES,
        "decode_settings": DECODE,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "subset_split": SELECTION_SPLIT,
        "subset_size": len(subset),
        "subset_speakers": sorted(subset["speaker_id"].unique().tolist()),
        "metrics": all_metrics,
        "selected_model": chosen,
        "selection_rule": select.__doc__,
        "reasoning": reasoning,
    }, indent=2))

    print("\n=== COMPARISON ===")
    print(comparison[["model", "wer", "exact_match_accuracy", "severe_count",
                      "severe_rate_of_all", "seconds_per_clip", "realtime_factor",
                      "peak_rss_mb"]].to_string(index=False))
    print(f"\nSELECTED: {chosen}\n{reasoning}")


if __name__ == "__main__":
    main()
