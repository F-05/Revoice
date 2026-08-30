"""Steps 2-4 -- run faster-whisper over TORGO and build the evaluation dataset.

    python scripts/transcribe_torgo.py            # resume / continue
    python scripts/transcribe_torgo.py --limit 20 # smoke test
    python scripts/transcribe_torgo.py --csv-only # rebuild the CSV from cache

Runs entirely offline: the model is called in-process, never through the
Revoice HTTP API. Every finished utterance is appended to a JSONL cache
immediately, so a crash costs at most one sample.

The ASR settings live in ``asr_config.py`` and mirror the Revoice backend.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_config import HF_DATASET_ID, ASRConfig  # noqa: E402
from utils import EVAL_DATA_DIR, PROCESSED_DIR, classify_utterance, normalize_text  # noqa: E402

CACHE_PATH = PROCESSED_DIR / "whisper_predictions.jsonl"
# Canonical dataset location. A copy is kept under evaluation/data/ because the
# evaluation scripts were written against that path first.
CSV_PATH = PROCESSED_DIR / "torgo_whisper.csv"
CSV_COPY_PATH = EVAL_DATA_DIR / "torgo_whisper.csv"


# --------------------------------------------------------------------------
# ASR
# --------------------------------------------------------------------------
def load_model(config: ASRConfig):
    from faster_whisper import WhisperModel

    print(f"Loading faster-whisper model {config.model_size!r} "
          f"(device={config.device}, compute_type={config.compute_type}) ...", flush=True)
    return WhisperModel(config.model_size, device=config.device,
                        compute_type=config.compute_type)


def segment_confidence(segments: list) -> float | None:
    """Duration-weighted mean of exp(avg_logprob).

    Same formula as ``TranscriptionResult.confidence`` in the Revoice backend,
    so the numbers here are comparable with what the API returns. It is a rough
    proxy, not a calibrated probability.
    """
    scored = [s for s in segments if getattr(s, "avg_logprob", None) is not None]
    if not scored:
        return None
    weights = [max(s.end - s.start, 1e-3) for s in scored]
    total = sum(weights)
    if total <= 0:
        return None
    value = sum(math.exp(s.avg_logprob) * w for s, w in zip(scored, weights)) / total
    return min(max(value, 0.0), 1.0)


def transcribe_bytes(model, audio_bytes: bytes, config: ASRConfig, tmp_dir: Path) -> dict:
    """Transcribe one in-memory WAV, mirroring the backend's call exactly."""
    # The backend hands faster-whisper a file path and lets it decode/resample
    # through PyAV. Do the same here rather than pre-resampling ourselves.
    tmp = tmp_dir / "clip.wav"
    tmp.write_bytes(audio_bytes)

    started = time.perf_counter()
    segments_iter, info = model.transcribe(str(tmp), **config.transcribe_kwargs())
    segments = list(segments_iter)
    elapsed = time.perf_counter() - started

    text = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
    word_probs = [w.probability for s in segments for w in (getattr(s, "words", None) or [])
                  if getattr(w, "probability", None) is not None]

    return {
        "whisper_transcript": text,
        "asr_confidence": segment_confidence(segments),
        "asr_min_word_probability": min(word_probs) if word_probs else None,
        "asr_mean_word_probability": sum(word_probs) / len(word_probs) if word_probs else None,
        "asr_num_segments": len(segments),
        "asr_detected_language": getattr(info, "language", None),
        "audio_duration_sec": float(getattr(info, "duration", 0.0)) or None,
        "asr_processing_sec": round(elapsed, 3),
    }


# --------------------------------------------------------------------------
# Dataset iteration
# --------------------------------------------------------------------------
def iter_samples(limit: int | None = None):
    """Yield (sample_id, split, index, text, audio_bytes, sample_rate)."""
    ds = load_dataset(HF_DATASET_ID)
    ds = ds.cast_column("audio", Audio(decode=False))
    for split in ds:
        data = ds[split]
        count = len(data) if limit is None else min(limit, len(data))
        for i in range(count):
            row = data[i]
            raw = row["audio"]["bytes"]
            with sf.SoundFile(io.BytesIO(raw)) as f:
                sample_rate = f.samplerate
            yield f"{split}-{i:04d}", split, i, row["text"], raw, sample_rate


def load_cache() -> dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    records: dict[str, dict] = {}
    for line in CACHE_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn last line from a hard kill
        records[record["sample_id"]] = record
    return records


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------
COLUMNS = [
    "sample_id", "split", "index",
    "ground_truth", "whisper_transcript",
    "ground_truth_normalized", "whisper_transcript_normalized",
    "utterance_type", "num_reference_words",
    "asr_confidence", "asr_min_word_probability", "asr_mean_word_probability",
    "asr_num_segments", "asr_detected_language",
    "audio_duration_sec", "sample_rate_hz", "asr_processing_sec",
]
# NOTE: there is deliberately no `speaker_id` column. The Hugging Face dataset
# exposes only `audio` and `text` -- no speaker field, no source filename. See
# the README; speaker IDs are NOT available and are never invented here.


def write_csv(records: dict[str, dict]) -> None:
    import pandas as pd

    rows = []
    for record in records.values():
        row = dict(record)
        row["ground_truth_normalized"] = normalize_text(row.get("ground_truth"))
        row["whisper_transcript_normalized"] = normalize_text(row.get("whisper_transcript"))
        row["num_reference_words"] = len(row["ground_truth_normalized"].split())
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.reindex(columns=COLUMNS)
    df = df.sort_values(["split", "index"]).reset_index(drop=True)
    for path in (CSV_PATH, CSV_COPY_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"\nWrote {len(df)} rows -> {path}")
    print(df["utterance_type"].value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N samples of each split")
    parser.add_argument("--csv-only", action="store_true",
                        help="rebuild the CSV from the cache without running ASR")
    parser.add_argument("--overwrite", action="store_true",
                        help="ignore the cache and re-transcribe everything")
    args = parser.parse_args()

    if args.csv_only:
        cache = load_cache()
        if not cache:
            sys.exit(f"No cached predictions at {CACHE_PATH}")
        write_csv(cache)
        return

    config = ASRConfig()
    print("ASR configuration (mirrors the Revoice backend):")
    for key, value in config.describe().items():
        print(f"  {key} = {value!r}")

    cache = {} if args.overwrite else load_cache()
    if args.overwrite and CACHE_PATH.exists():
        CACHE_PATH.unlink()
    if cache:
        print(f"Resuming: {len(cache)} samples already cached.")

    samples = list(iter_samples(args.limit))
    todo = [s for s in samples if s[0] not in cache]
    print(f"{len(samples)} samples total, {len(todo)} to transcribe.")

    if todo:
        model = load_model(config)
        with tempfile.TemporaryDirectory(prefix="torgo-asr-") as tmp_name:
            tmp_dir = Path(tmp_name)
            with CACHE_PATH.open("a") as cache_file:
                progress = tqdm(todo, unit="clip", dynamic_ncols=True)
                for sample_id, split, index, text, raw, sample_rate in progress:
                    progress.set_postfix_str(f"{split} {text[:28]}")
                    record = {
                        "sample_id": sample_id,
                        "split": split,
                        "index": index,
                        "ground_truth": text,
                        "utterance_type": classify_utterance(text),
                        "sample_rate_hz": sample_rate,
                    }
                    record.update(transcribe_bytes(model, raw, config, tmp_dir))
                    cache_file.write(json.dumps(record) + "\n")
                    cache_file.flush()  # survive a crash mid-run
                    cache[sample_id] = record

    write_csv(cache)


if __name__ == "__main__":
    main()
