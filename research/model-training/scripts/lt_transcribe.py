"""Step 9-11 -- transcribe with the selected ASR and build repair pairs.

    python scripts/lt_transcribe.py

Transcribes every dysarthric sentence (all splits) plus the control safety
sample, caching each prediction to JSONL immediately so an interrupted run
resumes rather than repeating work.

Writes data/large_torgo/asr_predictions.csv and data/large_torgo/repair_pairs.csv.
"""

from __future__ import annotations

import argparse
import json
import math
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
CACHE = LT_DATA / "asr_cache.jsonl"

DECODE = {
    "language": "en",
    "beam_size": 5,
    "condition_on_previous_text": False,
    "vad_filter": True,
    "word_timestamps": True,
}

# Splits that go through ASR. Isolated dysarthric words are held out entirely
# for a later experiment and are never transcribed here.
WANTED_SPLITS = ["train", "validation", "test", "control_test"]


def selected_model() -> str:
    path = LT_RESULTS / "asr_selection.json"
    if not path.exists():
        sys.exit("Run scripts/lt_asr_compare.py first.")
    return json.loads(path.read_text())["selected_model"]


def load_cache() -> dict[str, dict]:
    if not CACHE.exists():
        return {}
    out = {}
    for line in CACHE.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[record["sample_id"]] = record
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    model_size = selected_model()
    splits = pd.read_csv(LT_DATA / "splits.csv", keep_default_na=False, na_values=[""])
    work = splits[splits["split"].isin(WANTED_SPLITS)].copy()
    if args.limit:
        work = work.head(args.limit)
    print(f"ASR model: {model_size} | {len(work)} utterances to cover")
    print(work["split"].value_counts().to_string())

    cache = load_cache()
    todo = [s for s in work["sample_id"] if s not in cache]
    print(f"cached: {len(cache)} | to transcribe: {len(todo)}")

    if todo:
        from faster_whisper import WhisperModel
        print(f"Loading {model_size} ...", flush=True)
        model = WhisperModel(model_size, device="auto", compute_type="int8")
        truth = dict(zip(work["sample_id"], work["ground_truth"]))
        with tempfile.TemporaryDirectory(prefix="lt-asr-") as tmp:
            clip = Path(tmp) / "clip.wav"
            with CACHE.open("a") as handle:
                for sample_id, raw in tqdm(iter_audio(todo), total=len(todo),
                                           unit="clip", dynamic_ncols=True):
                    clip.write_bytes(raw)
                    started = time.perf_counter()
                    segments_iter, info = model.transcribe(str(clip), **DECODE)
                    segments = list(segments_iter)
                    elapsed = time.perf_counter() - started

                    text = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
                    scored = [s for s in segments if getattr(s, "avg_logprob", None) is not None]
                    confidence = None
                    if scored:
                        weights = [max(s.end - s.start, 1e-3) for s in scored]
                        total = sum(weights)
                        if total > 0:
                            confidence = min(max(sum(math.exp(s.avg_logprob) * w
                                                     for s, w in zip(scored, weights))
                                                 / total, 0.0), 1.0)
                    record = {
                        "sample_id": sample_id,
                        "ground_truth": truth[sample_id],
                        "asr_transcript": text,
                        "asr_confidence": confidence,
                        "asr_model": model_size,
                        "asr_seconds": round(elapsed, 3),
                        "audio_duration_sec": float(getattr(info, "duration", 0.0) or 0.0),
                    }
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    cache[sample_id] = record

    # --- assemble ------------------------------------------------------------
    preds = pd.DataFrame([cache[s] for s in work["sample_id"] if s in cache])
    df = work.merge(preds, on=["sample_id", "ground_truth"], how="inner")
    df["asr_transcript"] = df["asr_transcript"].fillna("")
    df["ground_truth_normalized"] = df["ground_truth"].map(normalize_text)
    df["asr_transcript_normalized"] = df["asr_transcript"].map(normalize_text)
    df["asr_correct"] = df["ground_truth_normalized"] == df["asr_transcript_normalized"]

    stats = [edit_stats(r, h) for r, h in
             zip(df["ground_truth_normalized"], df["asr_transcript_normalized"])]
    df["WER"] = [s["wer"] for s in stats]
    df["severity"] = ["CORRECT" if ok else severity(s, h) for ok, s, h in
                      zip(df["asr_correct"], stats, df["asr_transcript_normalized"])]
    df["repairability"] = ["CORRECT" if ok else repairability(s, h) for ok, s, h in
                           zip(df["asr_correct"], stats, df["asr_transcript_normalized"])]

    columns = ["sample_id", "speaker_id", "speaker_group", "sex", "split",
               "utterance_type", "prompt_key", "prompt_seen_in_train",
               "unseen_prompt_test", "ground_truth", "asr_transcript",
               "ground_truth_normalized", "asr_transcript_normalized",
               "asr_correct", "WER", "severity", "repairability",
               "asr_confidence", "asr_model", "audio_duration_sec"]
    df = df.reindex(columns=columns)
    df.to_csv(LT_DATA / "asr_predictions.csv", index=False)
    print(f"\nWrote {LT_DATA / 'asr_predictions.csv'} ({len(df)} rows)")

    # --- repair pairs --------------------------------------------------------
    # Training keeps correct ASR (so the model learns that changing nothing is
    # often right) plus HIGH/MEDIUM repairability errors. LOW-repairability
    # catastrophes are excluded from TRAINING ONLY -- they stay in validation
    # and test so the evaluation stays honest about the ceiling.
    df["repair_input"] = df["asr_transcript_normalized"]
    df["repair_target"] = df["ground_truth_normalized"]
    trainable = df["repairability"].isin(["CORRECT", "HIGH", "MEDIUM"])
    df["train_eligible"] = trainable & df["split"].eq("train")
    df.to_csv(LT_DATA / "repair_pairs.csv", index=False)
    print(f"Wrote {LT_DATA / 'repair_pairs.csv'}")

    print("\n=== ASR OUTCOME BY SPLIT ===")
    print(df.groupby("split").agg(
        n=("sample_id", "count"),
        correct=("asr_correct", "sum"),
        wer_mean=("WER", "mean")).to_string())
    print("\n=== REPAIRABILITY BY SPLIT ===")
    print(pd.crosstab(df["split"], df["repairability"]).to_string())
    train_rows = df[df["split"] == "train"]
    excluded = int((~trainable & df["split"].eq("train")).sum())
    print(f"\nTrain rows: {len(train_rows)} | eligible: {int(df['train_eligible'].sum())} "
          f"| excluded LOW-repairability: {excluded}")


if __name__ == "__main__":
    main()
