"""Step 1 -- load the Hugging Face dataset and report exactly what is in it.

Reports splits, sizes, columns, audio format/sample rate, and whether any
speaker identifier or original-file metadata exists. Nothing is inferred: if a
speaker column is absent it is reported as absent.

    python scripts/inspect_dataset.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter

import io

import soundfile as sf
from datasets import Audio, load_dataset

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from asr_config import HF_DATASET_ID  # noqa: E402
from utils import RESULTS_DIR, classify_utterance  # noqa: E402

# Column names that would carry speaker / provenance information if present.
SPEAKER_HINTS = ("speaker", "spk", "subject", "participant", "session", "id")
PATH_HINTS = ("path", "file", "filename", "source", "audio_path", "name")


def main() -> None:
    print(f"Loading {HF_DATASET_ID} ...", flush=True)
    ds = load_dataset(HF_DATASET_ID)
    # Decode audio ourselves with soundfile rather than letting `datasets` do
    # it through librosa: fewer dependencies, and it keeps the stored filename.
    ds = ds.cast_column("audio", Audio(decode=False))

    report: dict = {"dataset_id": HF_DATASET_ID, "splits": {}}

    print("\n=== SPLITS ===")
    for split, data in ds.items():
        print(f"  {split:<12} {len(data):>6} samples")
        report["splits"][split] = {"num_samples": len(data)}

    total = sum(len(d) for d in ds.values())
    report["total_samples"] = total
    print(f"  {'TOTAL':<12} {total:>6} samples")

    first_split = next(iter(ds))
    data = ds[first_split]

    print("\n=== COLUMNS / FEATURES ===")
    for name, feature in data.features.items():
        print(f"  {name}: {feature}")
    report["columns"] = list(data.features)
    report["features"] = {k: str(v) for k, v in data.features.items()}

    # --- audio format -----------------------------------------------------
    print("\n=== AUDIO ===")
    audio_cols = [n for n, f in data.features.items() if isinstance(f, Audio)]
    report["audio_columns"] = audio_cols
    if audio_cols:
        col = audio_cols[0]
        print(f"  audio column: {col!r}")
        print(f"  declared sampling_rate: {data.features[col].sampling_rate}")
        sample = data[0][col]
        print(f"  stored keys: {sorted(sample)}")
        raw = sample["bytes"]
        with sf.SoundFile(io.BytesIO(raw)) as f:
            print(f"  container/subtype: {f.format} / {f.subtype}")
            print(f"  actual sampling_rate: {f.samplerate} Hz, channels={f.channels}")
            print(f"  frames={len(f)}  duration={len(f) / f.samplerate:.2f}s")
            fmt, sr, ch = f.format, f.samplerate, f.channels
        rates = []
        for i in range(min(50, len(data))):
            with sf.SoundFile(io.BytesIO(data[i][col]["bytes"])) as f:
                rates.append(f.samplerate)
        print(f"  sample rates over first 50: {sorted(set(rates))}")
        print(f"  path field: {sample.get('path')!r}")
        report["audio"] = {
            "column": col,
            "declared_sampling_rate": data.features[col].sampling_rate,
            "file_format": fmt,
            "actual_sampling_rate": sr,
            "channels": ch,
            "sample_rates_first_50": sorted(set(rates)),
            "example_path": sample.get("path"),
        }
    else:
        print("  NO audio column found.")

    # --- speaker / provenance --------------------------------------------
    print("\n=== SPEAKER / PROVENANCE METADATA ===")
    speaker_cols = [c for c in data.features
                    if any(h in c.lower() for h in SPEAKER_HINTS) and c not in audio_cols]
    path_cols = [c for c in data.features
                 if any(h in c.lower() for h in PATH_HINTS) and c not in audio_cols]
    report["speaker_like_columns"] = speaker_cols
    report["path_like_columns"] = path_cols
    if speaker_cols:
        print(f"  speaker-like columns: {speaker_cols}")
        for c in speaker_cols:
            values = Counter(data[c])
            print(f"    {c}: {len(values)} distinct -> {list(values)[:10]}")
    else:
        print("  NO speaker-identifier column. Speaker IDs are NOT available.")
    if path_cols:
        print(f"  path-like columns: {path_cols}")
        for c in path_cols:
            print(f"    {c} examples: {data[c][:3]}")
    else:
        print("  NO original-filename/path column.")

    # Does the decoded audio dict carry a usable original path?
    if audio_cols:
        paths = [data[i][audio_cols[0]].get("path") for i in range(min(5, len(data)))]
        print(f"  audio['path'] first 5: {paths}")
        report["audio_paths_sample"] = paths

    # --- text -------------------------------------------------------------
    text_col = "text" if "text" in data.features else next(
        (c for c in data.features if data.features[c].__class__.__name__ == "Value"
         and c not in speaker_cols + path_cols), None)
    print(f"\n=== TEXT (column {text_col!r}) ===")
    if text_col:
        for split, d in ds.items():
            types = Counter(classify_utterance(t) for t in d[text_col])
            print(f"  {split}: word={types['word']} sentence={types['sentence']}")
            report["splits"][split]["utterance_types"] = dict(types)
        print("  examples:")
        for t in data[text_col][:10]:
            print(f"    {classify_utterance(t):<9} {t!r}")
    report["text_column"] = text_col

    out = RESULTS_DIR / "dataset_inspection.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
