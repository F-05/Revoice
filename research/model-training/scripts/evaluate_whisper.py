"""Steps 5-7 -- baseline faster-whisper evaluation on TORGO, with jiwer.

    python scripts/evaluate_whisper.py

Reads ``evaluation/data/torgo_whisper.csv`` and writes:
  results/baseline_metrics.json        machine-readable metrics
  evaluation/reports/baseline_report.md   metrics + error examples to read

Metrics are computed on the NORMALIZED columns only; the original
``ground_truth`` / ``whisper_transcript`` strings are never modified.

Word and sentence subsets are always reported separately. The sentence subset
is the one that matters for Revoice, because the repair model is meant to
exploit linguistic context that isolated words do not have.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jiwer
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_config import ASRConfig, HF_DATASET_ID  # noqa: E402
from utils import EVAL_DATA_DIR, EVAL_REPORT_DIR, RESULTS_DIR  # noqa: E402

CSV_PATH = EVAL_DATA_DIR / "torgo_whisper.csv"
METRICS_PATH = RESULTS_DIR / "baseline_metrics.json"
REPORT_PATH = EVAL_REPORT_DIR / "baseline_report.md"


def wer_stats(refs: list[str], hyps: list[str]) -> dict:
    """Corpus WER plus the raw edit-operation counts behind it."""
    if not refs:
        return {"num_utterances": 0}
    out = jiwer.process_words(refs, hyps)
    exact = sum(r == h for r, h in zip(refs, hyps))
    return {
        "num_utterances": len(refs),
        "wer": out.wer,
        "mer": out.mer,
        "wil": out.wil,
        "exact_match_accuracy": exact / len(refs),
        "num_exact_match": exact,
        "num_with_error": len(refs) - exact,
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "hits": out.hits,
        "num_reference_words": out.hits + out.substitutions + out.deletions,
        "num_empty_hypotheses": sum(1 for h in hyps if not h.strip()),
    }


def per_utterance_wer(df: pd.DataFrame) -> pd.Series:
    """WER of each row on its own. Can exceed 1.0 when Whisper over-generates."""
    return pd.Series(
        [jiwer.wer(r, h if h.strip() else "*") if r.strip() else float("nan")
         for r, h in zip(df["ground_truth_normalized"], df["whisper_transcript_normalized"])],
        index=df.index,
    )


def fmt_block(title: str, stats: dict) -> str:
    if not stats.get("num_utterances"):
        return f"### {title}\n\n_No utterances in this subset._\n"
    return (
        f"### {title}\n\n"
        f"| metric | value |\n|---|---|\n"
        f"| utterances | {stats['num_utterances']} |\n"
        f"| **WER** | **{stats['wer']:.4f}** ({stats['wer'] * 100:.2f}%) |\n"
        f"| match error rate (MER) | {stats['mer']:.4f} |\n"
        f"| word information lost (WIL) | {stats['wil']:.4f} |\n"
        f"| exact-match accuracy | {stats['exact_match_accuracy']:.4f} "
        f"({stats['exact_match_accuracy'] * 100:.2f}%) |\n"
        f"| perfectly transcribed | {stats['num_exact_match']} |\n"
        f"| with >=1 error | {stats['num_with_error']} |\n"
        f"| reference words | {stats['num_reference_words']} |\n"
        f"| substitutions / deletions / insertions | "
        f"{stats['substitutions']} / {stats['deletions']} / {stats['insertions']} |\n"
        f"| empty Whisper output | {stats['num_empty_hypotheses']} |\n"
    )


def examples_block(df: pd.DataFrame, title: str, n: int = 8) -> str:
    """Correct / small-error / severe-failure examples for one subset."""
    lines = [f"### {title}\n"]
    if df.empty:
        return lines[0] + "\n_none_\n"

    correct = df[df["utterance_wer"] == 0]
    small = df[(df["utterance_wer"] > 0) & (df["utterance_wer"] <= 0.5)]
    severe = df[df["utterance_wer"] > 0.5]

    for label, subset, order in (
        (f"Correct ({len(correct)})", correct, None),
        (f"Small errors, WER <= 0.5 ({len(small)})", small, "utterance_wer"),
        (f"Severe failures, WER > 0.5 ({len(severe)})", severe, "utterance_wer"),
    ):
        lines.append(f"\n**{label}**\n")
        if subset.empty:
            lines.append("\n_none_\n")
            continue
        picked = subset if order is None else subset.sort_values(order, ascending=False)
        for _, row in picked.head(n).iterrows():
            conf = row["asr_confidence"]
            conf_txt = "n/a" if pd.isna(conf) else f"{conf:.3f}"
            lines.append(
                f"\n- `{row['sample_id']}`  WER `{row['utterance_wer']:.2f}`  "
                f"conf `{conf_txt}`\n"
                f"  - truth:   `{row['ground_truth']}`\n"
                f"  - whisper: `{row['whisper_transcript'] if isinstance(row['whisper_transcript'], str) and row['whisper_transcript'] else '<empty>'}`\n"
            )
    return "".join(lines)


def main() -> None:
    if not CSV_PATH.exists():
        sys.exit(f"{CSV_PATH} not found -- run scripts/transcribe_torgo.py first.")

    df = pd.read_csv(CSV_PATH, keep_default_na=False, na_values=[""])
    df["ground_truth_normalized"] = df["ground_truth_normalized"].fillna("")
    df["whisper_transcript_normalized"] = df["whisper_transcript_normalized"].fillna("")

    unscorable = df[df["ground_truth_normalized"].str.strip() == ""]
    if not unscorable.empty:
        print(f"Skipping {len(unscorable)} rows with an empty reference.")
        df = df.drop(unscorable.index)

    df["utterance_wer"] = per_utterance_wer(df)

    words = df[df["utterance_type"] == "word"]
    sentences = df[df["utterance_type"] == "sentence"]

    # The upstream train/test split repeats prompts across sides, which is one
    # more reason not to trust it as an evaluation split (see README).
    train_texts = set(df[df["split"] == "train"]["ground_truth"])
    test_texts = set(df[df["split"] == "test"]["ground_truth"])

    metrics = {
        "dataset_id": HF_DATASET_ID,
        "split_leakage": {
            "unique_ground_truth_texts": int(df["ground_truth"].nunique()),
            "total_rows": len(df),
            "texts_in_both_splits": len(train_texts & test_texts),
            "test_texts_also_in_train": len(test_texts & train_texts),
            "num_test_texts": len(test_texts),
        },
        "asr_config": ASRConfig().describe(),
        "speaker_ids_available": False,  # verified in scripts/inspect_dataset.py
        "rows_skipped_empty_reference": len(unscorable),
        "split_counts": df["split"].value_counts().to_dict(),
        "overall": wer_stats(list(df["ground_truth_normalized"]),
                             list(df["whisper_transcript_normalized"])),
        "words": wer_stats(list(words["ground_truth_normalized"]),
                           list(words["whisper_transcript_normalized"])),
        "sentences": wer_stats(list(sentences["ground_truth_normalized"]),
                               list(sentences["whisper_transcript_normalized"])),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    # --- console summary --------------------------------------------------
    o, w, s = metrics["overall"], metrics["words"], metrics["sentences"]
    print("\n================ BASELINE faster-whisper on TORGO ================")
    print(f"ASR: {ASRConfig().model_size}  (mirrors Revoice backend)")
    print(f"\nA. OVERALL    n={o['num_utterances']:<5} WER={o['wer']:.4f}  "
          f"exact={o['exact_match_accuracy'] * 100:.2f}%")
    print(f"B. WORDS      n={w['num_utterances']:<5} WER={w['wer']:.4f}  "
          f"exact={w['exact_match_accuracy'] * 100:.2f}%")
    print(f"C. SENTENCES  n={s['num_utterances']:<5} WER={s['wer']:.4f}  "
          f"exact={s['exact_match_accuracy'] * 100:.2f}%")
    print(f"\nPerfect transcriptions: {o['num_exact_match']}/{o['num_utterances']} "
          f"({o['exact_match_accuracy'] * 100:.2f}%)")
    print(f"With >=1 error:         {o['num_with_error']}/{o['num_utterances']}")
    print(f"Empty Whisper output:   {o['num_empty_hypotheses']}")
    leak = metrics["split_leakage"]
    print(f"\nUpstream split: {leak['texts_in_both_splits']} prompt texts appear in "
          f"BOTH train and test ({leak['unique_ground_truth_texts']} unique texts "
          f"across {leak['total_rows']} rows). Do not treat it as a clean split.")

    # --- markdown report --------------------------------------------------
    report = [
        "# Baseline faster-whisper on TORGO (dysarthric male subset)\n",
        f"\nDataset: `{HF_DATASET_ID}`  \n",
        f"ASR config (copied from the Revoice backend): "
        f"`{json.dumps(ASRConfig().describe())}`\n",
        "\n> Metrics use the normalized text columns. The original "
        "`ground_truth` / `whisper_transcript` strings are unchanged in the CSV.\n",
        "\n> **No speaker IDs exist in this dataset**, so no speaker-disjoint "
        "split is possible and none is claimed. This is a male dysarthric "
        "subset, not all of TORGO and not all dysarthric speech.\n",
        f"\n> The upstream `train`/`test` split is **not clean**: "
        f"{metrics['split_leakage']['texts_in_both_splits']} prompt texts appear "
        f"on both sides, and the 770 rows contain only "
        f"{metrics['split_leakage']['unique_ground_truth_texts']} unique texts.\n",
        "\n## A. Overall\n\n", fmt_block("All utterances", o),
        "\n## B. Isolated words\n\n", fmt_block("utterance_type == word", w),
        "\n## C. Sentences (most relevant to Revoice)\n\n",
        fmt_block("utterance_type == sentence", s),
        "\n## Error examples\n",
        "\n", examples_block(sentences, "Sentences"),
        "\n", examples_block(words, "Isolated words"),
    ]
    REPORT_PATH.write_text("".join(report))
    print(f"\nWrote {METRICS_PATH}\nWrote {REPORT_PATH}")

    # Per-utterance WER back into a companion CSV for manual inspection.
    scored = EVAL_DATA_DIR / "torgo_whisper_scored.csv"
    df.to_csv(scored, index=False)
    print(f"Wrote {scored}")


if __name__ == "__main__":
    main()
