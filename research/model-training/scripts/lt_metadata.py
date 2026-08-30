"""Step 1-4 -- build verified metadata for the larger TORGO dataset.

    python scripts/lt_metadata.py

Reads the already-cached parquet files (no re-download, no audio decoding) and
writes:
  data/large_torgo/metadata.csv
  results/large_torgo/dataset_summary.json
  results/large_torgo/speaker_summary.csv
  results/large_torgo/prompt_frequency.csv

Every field is either taken from a real column or parsed from the stored
filename with the documented rule below. Nothing is inferred beyond that.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import classify_utterance, normalize_text  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
LT_DATA = PROJECT / "data" / "large_torgo"
LT_RESULTS = PROJECT / "results" / "large_torgo"
LT_DATA.mkdir(parents=True, exist_ok=True)
LT_RESULTS.mkdir(parents=True, exist_ok=True)

HF_CACHE = PROJECT / ".hf_cache" / "hub" / "datasets--abnerh--TORGO-database"

# --------------------------------------------------------------------------
# TORGO filename convention
# --------------------------------------------------------------------------
# Observed form:  FC01_1_arrayMic_0066.wav
#                 ^^^^ ^ ^^^^^^^^ ^^^^
#                 |    |  |        utterance number within the session
#                 |    |  microphone channel (arrayMic / headMic)
#                 |    session number
#                 speaker code
#
# Speaker code is <sex><group?><number>:
#   sex    M or F
#   group  a literal "C" marks a CONTROL speaker; its absence marks a
#          dysarthric speaker (TORGO's documented convention: F01/M01 are
#          dysarthric, FC01/MC01 are matched controls).
#   number two digits.
#
# The `speech_status` column is parsed independently and the two are
# cross-checked; any row where they disagree is reported, never silently kept.
FILENAME_RE = re.compile(
    r"^(?P<speaker>(?P<sex>[MF])(?P<control>C?)(?P<num>\d{2}))"
    r"_(?P<session>\d+)_(?P<mic>[A-Za-z]+)_(?P<utt>\d+)\.wav$"
)


def parse_filename(name: str) -> dict | None:
    match = FILENAME_RE.match(name or "")
    if not match:
        return None
    g = match.groupdict()
    return {
        "speaker_id": g["speaker"],
        "sex": "female" if g["sex"] == "F" else "male",
        "speaker_group_from_filename": "control" if g["control"] == "C" else "dysarthric",
        "session": int(g["session"]),
        "microphone": g["mic"],
        "utterance_number": int(g["utt"]),
    }


def load_rows() -> pd.DataFrame:
    files = sorted(HF_CACHE.rglob("*.parquet"))
    if not files:
        sys.exit(f"No cached parquet under {HF_CACHE}. Download the dataset first.")
    print(f"Reading {len(files)} cached parquet file(s) (metadata columns only) ...")
    frames = []
    for path in files:
        pf = pq.ParquetFile(path)
        table = pf.read(columns=["audio.path", "transcription", "speech_status",
                                 "gender", "duration"])
        df = table.to_pandas()
        # `audio.path` arrives as a struct column with a single `path` field.
        df["original_filename"] = df.iloc[:, 0].map(
            lambda v: v.get("path") if isinstance(v, dict) else v)
        df = df.drop(columns=[df.columns[0]])
        frames.append(df)
        print(f"  {path.name}: {len(df)} rows")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    raw = load_rows()
    print(f"\nTotal rows: {len(raw)}")

    parsed = raw["original_filename"].map(parse_filename)
    unparsed = raw[parsed.isna()]
    if not unparsed.empty:
        print(f"WARNING: {len(unparsed)} filenames did not match the TORGO pattern; "
              "they are kept with null speaker fields and excluded from speaker work.")
        print("  examples:", unparsed["original_filename"].head(5).tolist())

    meta = pd.DataFrame([p or {} for p in parsed])
    df = pd.concat([raw.reset_index(drop=True), meta], axis=1)

    df["ground_truth"] = df["transcription"]
    df["ground_truth_normalized"] = df["ground_truth"].map(normalize_text)
    df["utterance_type"] = df["ground_truth"].map(classify_utterance)
    df["prompt_key"] = df["ground_truth_normalized"]

    # --- cross-check filename group against the speech_status column --------
    status = df["speech_status"].astype(str).str.strip().str.lower()
    print("\nspeech_status values:", dict(Counter(status)))
    df["speaker_group_from_column"] = status.map(
        lambda v: "control" if v in {"healthy", "control"} else
                  ("dysarthric" if "dys" in v else None))
    both = df.dropna(subset=["speaker_group_from_filename", "speaker_group_from_column"])
    disagree = both[both["speaker_group_from_filename"] != both["speaker_group_from_column"]]
    print(f"filename-vs-column group disagreements: {len(disagree)}")
    if not disagree.empty:
        print(disagree[["original_filename", "speech_status",
                        "speaker_group_from_filename"]].head(10).to_string())
    # The column is the dataset's own label; use it, and keep the filename
    # derivation alongside so the disagreement count stays auditable.
    df["speaker_group"] = df["speaker_group_from_column"].fillna(
        df["speaker_group_from_filename"])

    # --- cross-check sex ----------------------------------------------------
    gender = df["gender"].astype(str).str.strip().str.lower()
    sex_disagree = df[(df["sex"].notna()) & (gender != df["sex"])]
    print(f"filename-vs-column sex disagreements: {len(sex_disagree)}")
    df["sex"] = df["sex"].fillna(gender.where(gender.isin(["male", "female"])))

    df["sample_id"] = [f"lt-{i:06d}" for i in range(len(df))]

    columns = [
        "sample_id", "original_filename", "speaker_id", "speaker_group", "sex",
        "session", "microphone", "utterance_number", "ground_truth",
        "ground_truth_normalized", "utterance_type", "prompt_key", "duration",
        "speech_status", "speaker_group_from_filename", "speaker_group_from_column",
    ]
    df = df.reindex(columns=columns)
    df.to_csv(LT_DATA / "metadata.csv", index=False)
    print(f"\nWrote {LT_DATA / 'metadata.csv'} ({len(df)} rows)")

    # --- microphone / duplicate investigation ------------------------------
    print("\n=== MICROPHONES ===")
    mic_counts = df["microphone"].value_counts()
    print(mic_counts.to_string())
    dup_key = ["speaker_id", "session", "utterance_number"]
    grouped = df.dropna(subset=dup_key).groupby(dup_key)["microphone"].nunique()
    multi_mic = int((grouped > 1).sum())
    print(f"\n(speaker, session, utterance_number) groups: {len(grouped)}")
    print(f"  ... recorded on >1 microphone: {multi_mic}")
    # Confirm the transcripts really are identical within such a group.
    same_text = df.dropna(subset=dup_key).groupby(dup_key)["ground_truth_normalized"].nunique()
    print(f"  groups whose transcript differs across mics: {int((same_text > 1).sum())}")

    # --- composition --------------------------------------------------------
    def composition(group: str) -> dict:
        sub = df[df["speaker_group"] == group]
        return {
            "rows": int(len(sub)),
            "speakers": int(sub["speaker_id"].nunique()),
            "speaker_ids": sorted(sub["speaker_id"].dropna().unique().tolist()),
            "words": int((sub["utterance_type"] == "word").sum()),
            "sentences": int((sub["utterance_type"] == "sentence").sum()),
        }

    summary = {
        "total_rows": int(len(df)),
        "unparsed_filenames": int(len(unparsed)),
        "microphones": mic_counts.to_dict(),
        "multi_microphone_utterance_groups": multi_mic,
        "group_column_disagreements": int(len(disagree)),
        "dysarthric": composition("dysarthric"),
        "control": composition("control"),
        "unique_prompts_all": int(df["prompt_key"].nunique()),
        "unique_sentence_prompts": int(
            df[df["utterance_type"] == "sentence"]["prompt_key"].nunique()),
        "unique_dysarthric_sentence_prompts": int(
            df[(df["utterance_type"] == "sentence")
               & (df["speaker_group"] == "dysarthric")]["prompt_key"].nunique()),
    }

    print("\n=== COMPOSITION ===")
    print(f"TOTAL rows: {summary['total_rows']}")
    for group in ("dysarthric", "control"):
        c = summary[group]
        print(f"\n{group.upper()}: {c['rows']} rows, {c['speakers']} speakers, "
              f"{c['words']} words, {c['sentences']} sentences")
        print(f"  speakers: {c['speaker_ids']}")
    print(f"\nunique sentence prompts (all): {summary['unique_sentence_prompts']}")
    print(f"unique dysarthric sentence prompts: {summary['unique_dysarthric_sentence_prompts']}")

    # --- per-speaker table --------------------------------------------------
    per_speaker = (
        df.dropna(subset=["speaker_id"])
        .groupby(["speaker_id", "speaker_group", "sex"], dropna=False)
        .apply(lambda g: pd.Series({
            "total_utterances": len(g),
            "words": int((g["utterance_type"] == "word").sum()),
            "sentences": int((g["utterance_type"] == "sentence").sum()),
            "unique_sentence_prompts": int(
                g[g["utterance_type"] == "sentence"]["prompt_key"].nunique()),
            "sessions": g["session"].nunique(),
            "microphones": "|".join(sorted(g["microphone"].dropna().unique())),
        }), include_groups=False)
        .reset_index()
        .sort_values(["speaker_group", "speaker_id"])
    )
    per_speaker.to_csv(LT_RESULTS / "speaker_summary.csv", index=False)
    print("\n=== PER SPEAKER ===")
    print(per_speaker.to_string(index=False))

    # --- prompt frequency ---------------------------------------------------
    sentences = df[df["utterance_type"] == "sentence"]
    freq = (
        sentences.groupby("prompt_key")
        .apply(lambda g: pd.Series({
            "occurrences": len(g),
            "distinct_speakers": g["speaker_id"].nunique(),
            "dysarthric_occurrences": int((g["speaker_group"] == "dysarthric").sum()),
            "control_occurrences": int((g["speaker_group"] == "control").sum()),
            "example_ground_truth": g["ground_truth"].iloc[0],
        }), include_groups=False)
        .reset_index()
        .sort_values("occurrences", ascending=False)
    )
    freq.to_csv(LT_RESULTS / "prompt_frequency.csv", index=False)
    summary["sentence_prompts_used_by_multiple_speakers"] = int(
        (freq["distinct_speakers"] > 1).sum())
    print(f"\nsentence prompts used by >1 speaker: "
          f"{summary['sentence_prompts_used_by_multiple_speakers']} / {len(freq)}")

    (LT_RESULTS / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {LT_RESULTS / 'dataset_summary.json'}")


if __name__ == "__main__":
    main()
