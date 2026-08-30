"""Per-speaker ASR difficulty, measured rather than assumed.

    python scripts/lt_speaker_difficulty.py

TORGO ships no dysarthria-severity field, so this project never claims one.
What it can do honestly is measure how hard each speaker is FOR THE SELECTED
ASR, and check whether that difficulty is balanced across the splits. If the
test speakers are much easier or harder than the training speakers, the
headline test number has to be read in that light.

Writes results/large_torgo/speaker_difficulty.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jiwer
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT = Path(__file__).resolve().parent.parent
LT_DATA = PROJECT / "data" / "large_torgo"
LT_RESULTS = PROJECT / "results" / "large_torgo"


def corpus_wer(frame: pd.DataFrame) -> float:
    return jiwer.process_words(
        list(frame["ground_truth_normalized"]),
        [h if str(h).strip() else "*" for h in frame["asr_transcript_normalized"]]).wer


def main() -> None:
    df = pd.read_csv(LT_DATA / "asr_predictions.csv", keep_default_na=False, na_values=[""])
    df["asr_transcript_normalized"] = df["asr_transcript_normalized"].fillna("")

    rows = []
    for (speaker, group, sex, split), g in df.groupby(
            ["speaker_id", "speaker_group", "sex", "split"]):
        rows.append({
            "speaker_id": speaker,
            "speaker_group": group,
            "sex": sex,
            "split": split,
            "n": len(g),
            "asr_wer": corpus_wer(g),
            "asr_exact_match": float(g["asr_correct"].mean()),
            "severe_rate": float((g["severity"] == "SEVERE").mean()),
            "high_or_medium_repairable_rate": float(
                g["repairability"].isin(["HIGH", "MEDIUM"]).mean()),
        })
    table = pd.DataFrame(rows).sort_values(["speaker_group", "split", "asr_wer"])
    table.to_csv(LT_RESULTS / "speaker_difficulty.csv", index=False)

    print("=== PER-SPEAKER ASR DIFFICULTY (selected model) ===")
    print(table.to_string(index=False))

    dys = table[table["speaker_group"] == "dysarthric"]
    print("\n=== DIFFICULTY BY SPLIT (dysarthric sentences) ===")
    per_split = df[df["speaker_group"] == "dysarthric"].groupby("split")
    for split, g in per_split:
        print(f"  {split:<12} n={len(g):<4} WER={corpus_wer(g):.4f} "
              f"exact={g['asr_correct'].mean() * 100:5.2f}% "
              f"severe={(g['severity'] == 'SEVERE').mean() * 100:5.1f}%")
    spread = dys["asr_wer"].max() - dys["asr_wer"].min()
    print(f"\nper-speaker WER spread across dysarthric speakers: {spread:.3f} "
          f"({dys.loc[dys['asr_wer'].idxmin(), 'speaker_id']} easiest, "
          f"{dys.loc[dys['asr_wer'].idxmax(), 'speaker_id']} hardest)")
    print("\nNOTE: this is ASR difficulty, not a clinical severity rating. TORGO's "
          "severity labels are not present in this dataset and are never inferred here.")


if __name__ == "__main__":
    main()
