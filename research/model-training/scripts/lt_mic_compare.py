"""Does the microphone change the ASR errors, or only the audio?

    python scripts/lt_mic_compare.py

Identical ground-truth text across `arrayMic` and `headMic` is expected -- it is
one spoken utterance recorded twice. That alone is no reason to discard a
channel. The question that matters is whether the two channels produce
DIFFERENT ASR OUTPUT. If they do, arrayMic rows are potentially useful as extra
noisy-input variants for repair training; if they are near-identical, keeping
them would just duplicate training targets.

Runs on paired utterances from TRAIN and VALIDATION speakers only -- test
speakers are not read.

Writes results/large_torgo/mic_comparison.csv and mic_comparison_summary.json.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import jiwer
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_triage import edit_stats, severity  # noqa: E402
from lt_audio import iter_audio  # noqa: E402
from utils import normalize_text  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
LT_DATA = PROJECT / "data" / "large_torgo"
LT_RESULTS = PROJECT / "results" / "large_torgo"

DECODE = {
    "language": "en",
    "beam_size": 5,
    "condition_on_previous_text": False,
    "vad_filter": True,
    "word_timestamps": True,
}
PAIRS_PER_SPEAKER = 20
RANDOM_SEED = 20260829
# Speakers whose data is allowed to be inspected before the final evaluation.
ALLOWED_SPEAKERS_SPLITS = ("train", "validation")


def selected_model() -> str:
    path = LT_RESULTS / "asr_selection.json"
    if not path.exists():
        sys.exit("Run scripts/lt_asr_compare.py first.")
    return json.loads(path.read_text())["selected_model"]


def build_pairs() -> pd.DataFrame:
    """Utterances present on BOTH channels, from train/validation speakers."""
    meta = pd.read_csv(LT_DATA / "metadata.csv", keep_default_na=False, na_values=[""])
    splits = pd.read_csv(LT_DATA / "splits.csv", keep_default_na=False, na_values=[""])
    allowed = splits[splits["split"].isin(ALLOWED_SPEAKERS_SPLITS)]["speaker_id"].unique()

    sentences = meta[(meta["utterance_type"] == "sentence")
                     & (meta["speaker_group"] == "dysarthric")
                     & (meta["speaker_id"].isin(allowed))]

    key = ["speaker_id", "session", "utterance_number"]
    counts = sentences.groupby(key)["microphone"].nunique()
    paired_keys = counts[counts == 2].index
    paired = sentences.set_index(key).loc[paired_keys].reset_index()

    # A fixed number of utterance PAIRS per speaker, not per row.
    chosen = []
    for speaker, group in paired.groupby("speaker_id"):
        utterances = group[key].drop_duplicates()
        picked = utterances.sample(min(len(utterances), PAIRS_PER_SPEAKER),
                                   random_state=RANDOM_SEED)
        chosen.append(group.merge(picked, on=key))
    return pd.concat(chosen, ignore_index=True)


def main() -> None:
    model_size = selected_model()
    pairs = build_pairs()
    n_pairs = len(pairs.drop_duplicates(["speaker_id", "session", "utterance_number"]))
    print(f"ASR model: {model_size}")
    print(f"{n_pairs} paired utterances ({len(pairs)} recordings) from "
          f"{pairs['speaker_id'].nunique()} train/validation speakers")

    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="auto", compute_type="int8")

    ids = list(pairs["sample_id"])
    rows = []
    with tempfile.TemporaryDirectory(prefix="lt-mic-") as tmp:
        clip = Path(tmp) / "clip.wav"
        for sample_id, raw in tqdm(iter_audio(ids), total=len(ids), unit="clip",
                                   dynamic_ncols=True):
            clip.write_bytes(raw)
            segments, _ = model.transcribe(str(clip), **DECODE)
            text = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
            rows.append({"sample_id": sample_id, "asr_transcript": text})

    got = pd.DataFrame(rows)
    df = pairs.merge(got, on="sample_id")
    df["asr_normalized"] = df["asr_transcript"].map(normalize_text)
    df["truth_normalized"] = df["ground_truth"].map(normalize_text)
    stats = [edit_stats(r, h) for r, h in zip(df["truth_normalized"], df["asr_normalized"])]
    df["wer"] = [s["wer"] for s in stats]
    df["severity"] = ["CORRECT" if r == h else severity(s, h) for r, h, s in
                      zip(df["truth_normalized"], df["asr_normalized"], stats)]
    df.to_csv(LT_RESULTS / "mic_comparison.csv", index=False)

    # --- per-microphone quality --------------------------------------------
    per_mic = {}
    for mic, group in df.groupby("microphone"):
        out = jiwer.process_words(
            list(group["truth_normalized"]),
            [h if h.strip() else "*" for h in group["asr_normalized"]])
        per_mic[mic] = {
            "recordings": int(len(group)),
            "wer": out.wer,
            "mean_utterance_wer": float(group["wer"].mean()),
            "exact_match_accuracy": float(
                (group["truth_normalized"] == group["asr_normalized"]).mean()),
            "severe_count": int((group["severity"] == "SEVERE").sum()),
            "severe_rate": float((group["severity"] == "SEVERE").mean()),
            "correct_count": int((group["severity"] == "CORRECT").sum()),
        }

    # --- paired agreement ----------------------------------------------------
    key = ["speaker_id", "session", "utterance_number"]
    wide = df.pivot_table(index=key, columns="microphone",
                          values="asr_normalized", aggfunc="first").dropna()
    wide_wer = df.pivot_table(index=key, columns="microphone",
                              values="wer", aggfunc="first").dropna()
    wide_sev = df.pivot_table(index=key, columns="microphone",
                              values="severity", aggfunc="first").dropna()

    identical = int((wide["arrayMic"] == wide["headMic"]).sum())
    total = int(len(wide))
    cross_wer = jiwer.process_words(
        [h if h.strip() else "*" for h in wide["headMic"]],
        [a if a.strip() else "*" for a in wide["arrayMic"]]).wer

    both_correct = int(((wide_sev["arrayMic"] == "CORRECT")
                        & (wide_sev["headMic"] == "CORRECT")).sum())
    only_head = int(((wide_sev["headMic"] == "CORRECT")
                     & (wide_sev["arrayMic"] != "CORRECT")).sum())
    only_array = int(((wide_sev["arrayMic"] == "CORRECT")
                      & (wide_sev["headMic"] != "CORRECT")).sum())

    summary = {
        "asr_model": model_size,
        "paired_utterances": total,
        "speakers": sorted(df["speaker_id"].unique().tolist()),
        "speakers_source_splits": list(ALLOWED_SPEAKERS_SPLITS),
        "asr_transcripts_identical": identical,
        "asr_transcripts_identical_pct": identical / total * 100 if total else None,
        "asr_transcripts_differ": total - identical,
        "asr_transcripts_differ_pct": (total - identical) / total * 100 if total else None,
        "cross_microphone_wer": cross_wer,
        "per_microphone": per_mic,
        "both_channels_correct": both_correct,
        "only_headMic_correct": only_head,
        "only_arrayMic_correct": only_array,
    }
    (LT_RESULTS / "mic_comparison_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== PER MICROPHONE ===")
    print(pd.DataFrame(per_mic).T.to_string())
    print("\n=== PAIRED AGREEMENT ===")
    print(f"paired utterances: {total}")
    print(f"  ASR output identical: {identical} ({identical / total:.1%})")
    print(f"  ASR output differs:   {total - identical} ({(total - identical) / total:.1%})")
    print(f"  cross-microphone WER (headMic as reference): {cross_wer:.4f}")
    print(f"  both channels correct: {both_correct}")
    print(f"  only headMic correct:  {only_head}")
    print(f"  only arrayMic correct: {only_array}")
    print(f"\nWrote {LT_RESULTS / 'mic_comparison_summary.json'}")


if __name__ == "__main__":
    main()
