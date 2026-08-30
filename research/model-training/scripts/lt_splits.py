"""Steps 3, 5, 6 -- microphone selection, speaker-disjoint split, prompt leakage.

    python scripts/lt_splits.py

Writes data/large_torgo/splits.csv and results/large_torgo/split_summary.json.

MICROPHONE DECISION
-------------------
TORGO records each utterance simultaneously on two channels, `arrayMic` (room
array) and `headMic` (close-talking). 7,477 of 9,075 (speaker, session,
utterance) groups appear on both. Their transcripts are identical, which is
expected and says nothing on its own -- it is the same spoken utterance, so of
course the target text matches.

Identical targets are therefore NOT the reason to drop a channel. Since the
split is speaker-disjoint, a microphone variant cannot cross splits anyway: both
channels of an utterance belong to one speaker, and that speaker sits in exactly
one split. The real concerns are:
  - duplicate linguistic targets (the same sentence counted twice),
  - an inflated apparent sample count,
  - statistically correlated examples inside a split.

This experiment uses **headMic only** as the primary condition, for reasons that
are about acoustics and bookkeeping rather than leakage:
  - it is the close-talking channel, the cleaner and more standard choice;
  - it yields more dysarthric sentences than arrayMic (683 vs 638);
  - a single channel keeps ASR comparisons unconfounded by microphone.

Whether arrayMic is worth keeping as an extra noisy-input variant for repair
training is an empirical question, answered separately by
scripts/lt_mic_compare.py. Either way, prompt counts in the dataset statistics
are computed over unique normalized prompts, so a second channel never inflates
them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT = Path(__file__).resolve().parent.parent
LT_DATA = PROJECT / "data" / "large_torgo"
LT_RESULTS = PROJECT / "results" / "large_torgo"

MICROPHONE = "headMic"

# Speaker-disjoint assignment. Chosen by hand so that every split holds at least
# one female and one male dysarthric speaker, the training side keeps the
# largest speakers, and test holds two speakers never seen in training.
SPEAKER_SPLITS = {
    "train": ["F03", "M01", "M02", "M03"],
    "validation": ["F01", "M04"],
    "test": ["F04", "M05"],
}

# Control speakers are only ever used for the safety evaluation, never for
# training. A fixed sample keeps transcription time reasonable.
CONTROL_TEST_SAMPLES = 200
RANDOM_SEED = 20260829


def main() -> None:
    meta = pd.read_csv(LT_DATA / "metadata.csv", keep_default_na=False, na_values=[""])
    meta = meta[meta["microphone"] == MICROPHONE].copy()
    print(f"Using microphone={MICROPHONE!r}: {len(meta)} rows "
          f"(of {len(pd.read_csv(LT_DATA / 'metadata.csv'))} total)")

    assigned = {s: split for split, speakers in SPEAKER_SPLITS.items() for s in speakers}
    overlap = [s for s in assigned if sum(s in v for v in SPEAKER_SPLITS.values()) > 1]
    assert not overlap, f"speaker in more than one split: {overlap}"

    meta["split"] = meta["speaker_id"].map(assigned)

    dys = meta[meta["speaker_group"] == "dysarthric"].copy()
    dys_sent = dys[dys["utterance_type"] == "sentence"].copy()
    dys_word = dys[dys["utterance_type"] == "word"].copy()
    ctl_sent = meta[(meta["speaker_group"] == "control")
                    & (meta["utterance_type"] == "sentence")].copy()

    missing = dys_sent[dys_sent["split"].isna()]
    assert missing.empty, f"unassigned dysarthric speakers: {missing.speaker_id.unique()}"

    # --- control safety sample ---------------------------------------------
    ctl_sent["split"] = "control_test"
    per_speaker = max(1, CONTROL_TEST_SAMPLES // ctl_sent["speaker_id"].nunique())
    ctl_sample = pd.concat(
        [g.sample(min(len(g), per_speaker), random_state=RANDOM_SEED)
         for _, g in ctl_sent.groupby("speaker_id")],
        ignore_index=True,
    )
    print(f"\nControl safety sample: {len(ctl_sample)} sentences from "
          f"{ctl_sample['speaker_id'].nunique()} control speakers "
          f"(<= {per_speaker} each, seed {RANDOM_SEED}).")

    # --- prompt leakage ------------------------------------------------------
    train_prompts = set(dys_sent[dys_sent["split"] == "train"]["prompt_key"])
    val_prompts = set(dys_sent[dys_sent["split"] == "validation"]["prompt_key"])
    test_prompts = set(dys_sent[dys_sent["split"] == "test"]["prompt_key"])

    dys_sent["prompt_seen_in_train"] = dys_sent["prompt_key"].isin(train_prompts)
    # The stricter generalisation subset: test speakers AND prompts absent from
    # training. Small by construction -- that is expected, not a defect.
    unseen = dys_sent[(dys_sent["split"] == "test") & (~dys_sent["prompt_seen_in_train"])]
    dys_sent["unseen_prompt_test"] = dys_sent.index.isin(unseen.index)

    leakage = {
        "train_prompts": len(train_prompts),
        "validation_prompts": len(val_prompts),
        "test_prompts": len(test_prompts),
        "test_prompts_also_in_train": len(test_prompts & train_prompts),
        "validation_prompts_also_in_train": len(val_prompts & train_prompts),
        "test_rows_with_prompt_seen_in_train": int(
            dys_sent[(dys_sent["split"] == "test")]["prompt_seen_in_train"].sum()),
        "unseen_prompt_test_rows": int(len(unseen)),
        "unseen_prompt_test_prompts": int(unseen["prompt_key"].nunique()),
    }

    # --- write ---------------------------------------------------------------
    keep = ["sample_id", "original_filename", "speaker_id", "speaker_group", "sex",
            "session", "microphone", "utterance_number", "ground_truth",
            "ground_truth_normalized", "utterance_type", "prompt_key", "duration",
            "split", "prompt_seen_in_train", "unseen_prompt_test"]
    dys_word["split"] = "dysarthric_words_holdout"
    for frame in (dys_word, ctl_sample):
        frame["prompt_seen_in_train"] = frame["prompt_key"].isin(train_prompts)
        frame["unseen_prompt_test"] = False

    splits = pd.concat([dys_sent, dys_word, ctl_sample], ignore_index=True)[keep]
    splits.to_csv(LT_DATA / "splits.csv", index=False)

    # --- report ---------------------------------------------------------------
    print("\n=== SPEAKER-DISJOINT SPLIT (dysarthric sentences, headMic) ===")
    table = (dys_sent.groupby(["split", "speaker_id", "sex"])
             .size().rename("sentences").reset_index())
    print(table.to_string(index=False))
    print("\nper split:")
    per_split = dys_sent.groupby("split").agg(
        sentences=("sample_id", "count"),
        speakers=("speaker_id", "nunique"),
        prompts=("prompt_key", "nunique")).reset_index()
    print(per_split.to_string(index=False))

    print("\n=== PROMPT LEAKAGE ===")
    for k, v in leakage.items():
        print(f"  {k}: {v}")

    summary = {
        "microphone_used": MICROPHONE,
        "microphone_rationale": "identical transcripts on both channels for 7477/9075 "
                                "utterance groups; headMic is close-talking and yields "
                                "more dysarthric sentences (683 vs 638)",
        "speaker_splits": SPEAKER_SPLITS,
        "dysarthric_sentences": {
            "total": int(len(dys_sent)),
            "per_split": dys_sent.groupby("split").size().to_dict(),
            "per_speaker": dys_sent.groupby("speaker_id").size().to_dict(),
        },
        "dysarthric_words_holdout": int(len(dys_word)),
        "control_safety_sample": {
            "rows": int(len(ctl_sample)),
            "speakers": sorted(ctl_sample["speaker_id"].unique().tolist()),
            "seed": RANDOM_SEED,
        },
        "prompt_leakage": leakage,
    }
    (LT_RESULTS / "split_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {LT_DATA / 'splits.csv'} ({len(splits)} rows)")
    print(f"Wrote {LT_RESULTS / 'split_summary.json'}")


if __name__ == "__main__":
    main()
