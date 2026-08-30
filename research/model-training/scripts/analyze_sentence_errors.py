"""Baseline analysis: per-group WER, sentence error triage, and the report.

    python scripts/analyze_sentence_errors.py

Reads the cached predictions (no ASR is run) and writes:
  data/processed/torgo_whisper.csv   the dataset (rebuilt from cache)
  results/sentence_errors.csv        every incorrect sentence, triaged
  results/baseline_metrics.json      machine-readable metrics
  results/baseline_report.md         human-readable report
  results/failed_samples.csv         samples where ASR produced nothing

Severity and repairability below are DETERMINISTIC ANALYSIS HEURISTICS chosen
for triage on this dataset. They are not clinically or scientifically validated
categories, and `repairability` is an estimate of whether a text-only repair
model could plausibly have enough context -- not evidence that any sentence
actually can be repaired. Only a trained-and-evaluated model can show that.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jiwer
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_config import ASRConfig, HF_DATASET_ID  # noqa: E402
from error_triage import (  # noqa: E402
    REPAIRABILITY_RULE, SEVERITY_RULE, edit_profile, edit_stats, repair_reason,
    repairability, severity,
)
from utils import PROCESSED_DIR, RESULTS_DIR  # noqa: E402

CSV_PATH = PROCESSED_DIR / "torgo_whisper.csv"
ERRORS_PATH = RESULTS_DIR / "sentence_errors.csv"
METRICS_PATH = RESULTS_DIR / "baseline_metrics.json"
REPORT_PATH = RESULTS_DIR / "baseline_report.md"
FAILED_PATH = RESULTS_DIR / "failed_samples.csv"


# Severity / repairability heuristics now live in scripts/error_triage.py so
# that this experiment and the larger TORGO experiment use an identical rule.
# --------------------------------------------------------------------------
# Group metrics
# --------------------------------------------------------------------------
def group_metrics(df: pd.DataFrame) -> dict:
    refs = list(df["ground_truth_normalized"])
    hyps = list(df["whisper_transcript_normalized"])
    if not refs:
        return {"num_samples": 0}
    out = jiwer.process_words(refs, [h if h.strip() else "*" for h in hyps])
    exact = int((df["ground_truth_normalized"] == df["whisper_transcript_normalized"]).sum())
    return {
        "num_samples": len(refs),
        "wer": out.wer,
        "exact_match_accuracy": exact / len(refs),
        "num_exact_match": exact,
        "num_with_error": len(refs) - exact,
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "hits": out.hits,
        "reference_words": out.hits + out.substitutions + out.deletions,
        "num_empty_hypotheses": int((df["whisper_transcript_normalized"].str.strip() == "").sum()),
    }


def counts_with_pct(series: pd.Series, order: list[str]) -> dict:
    total = len(series)
    return {k: {"count": int((series == k).sum()),
                "pct": (float((series == k).sum()) / total * 100) if total else 0.0}
            for k in order}


def main() -> None:
    df = pd.read_csv(CSV_PATH, keep_default_na=False, na_values=[""])
    for col in ("ground_truth_normalized", "whisper_transcript_normalized",
                "whisper_transcript"):
        df[col] = df[col].fillna("")

    # --- failures ---------------------------------------------------------
    failed = df[df["whisper_transcript"].str.strip() == ""].copy()
    if not failed.empty:
        failed["failure"] = "empty ASR output"
        failed["reason"] = failed.apply(
            lambda r: ("vad_filter removed all audio: transcribe() returned 0 segments"
                       if r["asr_num_segments"] == 0 else "ASR returned only empty segments"),
            axis=1)
        failed[["sample_id", "split", "ground_truth", "audio_duration_sec",
                "asr_num_segments", "failure", "reason"]].to_csv(FAILED_PATH, index=False)

    # --- per-utterance stats ---------------------------------------------
    stats = [edit_stats(r, h) for r, h in
             zip(df["ground_truth_normalized"], df["whisper_transcript_normalized"])]
    for key in ("wer", "hits", "substitutions", "deletions", "insertions",
                "total_errors", "ref_words", "hyp_words", "content_preserved",
                "length_ratio"):
        df[key if key != "wer" else "utterance_wer"] = [s[key] for s in stats]

    words = df[df["utterance_type"] == "word"]
    sentences = df[df["utterance_type"] == "sentence"].copy()

    # --- sentence errors --------------------------------------------------
    errors = sentences[
        sentences["ground_truth_normalized"] != sentences["whisper_transcript_normalized"]
    ].copy()
    err_stats = [edit_stats(r, h) for r, h in
                 zip(errors["ground_truth_normalized"], errors["whisper_transcript_normalized"])]
    errors["error_category"] = [severity(s, h) for s, h in
                                zip(err_stats, errors["whisper_transcript_normalized"])]
    errors["repairability"] = [repairability(s, h) for s, h in
                               zip(err_stats, errors["whisper_transcript_normalized"])]
    errors["edit_profile"] = [edit_profile(s, h) for s, h in
                              zip(err_stats, errors["whisper_transcript_normalized"])]
    errors["repairability_reason"] = [
        repair_reason(sev, rep, s, h) for sev, rep, s, h in
        zip(errors["error_category"], errors["repairability"], err_stats,
            errors["whisper_transcript_normalized"])]
    errors = errors.rename(columns={"split": "original_split", "utterance_wer": "WER"})

    error_cols = [
        "sample_id", "original_split", "ground_truth", "whisper_transcript",
        "ground_truth_normalized", "whisper_transcript_normalized",
        "WER", "error_category", "repairability", "edit_profile",
        "hits", "substitutions", "deletions", "insertions", "total_errors",
        "ref_words", "hyp_words", "content_preserved", "length_ratio",
        # Genuine engine-reported values, carried through unchanged. Blank where
        # the engine reported none (never invented).
        "asr_confidence", "asr_min_word_probability", "asr_mean_word_probability",
        "repairability_reason",
    ]
    errors[error_cols].to_csv(ERRORS_PATH, index=False)

    sev_counts = counts_with_pct(errors["error_category"], ["MINOR", "MODERATE", "SEVERE"])
    rep_counts = counts_with_pct(errors["repairability"], ["HIGH", "MEDIUM", "LOW"])

    # --- split composition ------------------------------------------------
    composition = {
        split: {
            "total": int(len(g)),
            "words": int((g["utterance_type"] == "word").sum()),
            "sentences": int((g["utterance_type"] == "sentence").sum()),
        }
        for split, g in df.groupby("split")
    }

    metrics = {
        "dataset_id": HF_DATASET_ID,
        "asr_config": ASRConfig().describe(),
        "faster_whisper_version": "1.2.1",
        "speaker_ids_available": False,
        "total_samples": len(df),
        "overall": group_metrics(df),
        "words": group_metrics(words),
        "sentences": group_metrics(sentences),
        "sentence_error_severity": sev_counts,
        "sentence_error_repairability": rep_counts,
        "split_composition": composition,
        "failed_samples": (failed[["sample_id", "ground_truth"]].to_dict("records")
                           if not failed.empty else []),
        "heuristics": {"severity": SEVERITY_RULE, "repairability": REPAIRABILITY_RULE},
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    write_report(metrics, errors)

    # --- console ----------------------------------------------------------
    o, w, s = metrics["overall"], metrics["words"], metrics["sentences"]
    for name, m in (("TOTAL", o), ("WORDS", w), ("SENTENCES", s)):
        print(f"{name:<10} n={m['num_samples']:<4} WER={m['wer']:.4f}  "
              f"exact={m['exact_match_accuracy'] * 100:.2f}%")
    print("\nSeverity:", {k: v["count"] for k, v in sev_counts.items()})
    print("Repairability:", {k: v["count"] for k, v in rep_counts.items()})
    print("Split composition:", composition)
    for path in (CSV_PATH, ERRORS_PATH, METRICS_PATH, REPORT_PATH):
        print("Wrote", path)
    if not failed.empty:
        print("Wrote", FAILED_PATH, f"({len(failed)} sample(s))")


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def example_block(row: pd.Series) -> str:
    hyp = row["whisper_transcript"] if str(row["whisper_transcript"]).strip() else "<empty output>"
    conf = row["asr_confidence"]
    conf_line = ("not reported by the engine" if pd.isna(conf) else f"{conf:.3f}")
    return (
        f"\n`{row['sample_id']}` ({row['original_split']} split)\n\n"
        f"```\n"
        f"Ground truth:\n\"{row['ground_truth']}\"\n\n"
        f"Whisper:\n\"{hyp}\"\n\n"
        f"WER:\n{row['WER']:.2f}\n\n"
        f"Severity:\n{row['error_category'].title()}\n\n"
        f"Estimated repairability:\n{row['repairability'].title()}\n\n"
        f"ASR confidence (engine-reported):\n{conf_line}\n\n"
        f"Reason:\n{row['repairability_reason']}\n"
        f"```\n"
    )


def write_report(metrics: dict, errors: pd.DataFrame) -> None:
    o, w, s = metrics["overall"], metrics["words"], metrics["sentences"]
    sev, rep = metrics["sentence_error_severity"], metrics["sentence_error_repairability"]
    comp = metrics["split_composition"]

    lines = [
        "# Baseline: faster-whisper on TORGO (dysarthric male subset)\n",
        f"\nDataset: `{metrics['dataset_id']}`  \n",
        f"Engine: faster-whisper {metrics['faster_whisper_version']}, "
        f"config `{json.dumps(metrics['asr_config'])}`\n",
        "\nThis is a single, unchanged baseline configuration matching the Revoice "
        "backend. No alternative ASR settings were run.\n",
        "\n## Summary\n\n",
        "| | Samples | WER | Exact match |\n|---|---|---|---|\n",
        f"| Overall | {o['num_samples']} | {o['wer']:.4f} | "
        f"{o['exact_match_accuracy'] * 100:.2f}% |\n",
        f"| Words | {w['num_samples']} | {w['wer']:.4f} | "
        f"{w['exact_match_accuracy'] * 100:.2f}% |\n",
        f"| Sentences | {s['num_samples']} | {s['wer']:.4f} | "
        f"{s['exact_match_accuracy'] * 100:.2f}% |\n",
        f"\nWord WER exceeds 1.0 because Whisper inserts words: across the whole set "
        f"there are {o['insertions']} insertions against {o['reference_words']} reference "
        f"words. For isolated words, **exact-match accuracy "
        f"({w['exact_match_accuracy'] * 100:.2f}%) is the meaningful number**, not WER.\n",
        "\n## Sentence results (primary metric for Revoice)\n\n",
        f"- total sentence samples: **{s['num_samples']}**\n",
        f"- correct sentences: **{s['num_exact_match']}**\n",
        f"- incorrect sentences: **{s['num_with_error']}**\n",
        f"- sentence WER: **{s['wer']:.4f}** ({s['wer'] * 100:.2f}%)\n",
        f"- exact-match accuracy: **{s['exact_match_accuracy'] * 100:.2f}%**\n",
        f"- substitutions / deletions / insertions: "
        f"{s['substitutions']} / {s['deletions']} / {s['insertions']}\n",
        "\nSentences are treated as the primary result because the proposed repair "
        "model depends on linguistic context, which isolated words do not provide.\n",
        "\n## Error severity\n\n",
        f"Of the {s['num_with_error']} incorrect sentences:\n\n",
        "| severity | count | % of incorrect sentences |\n|---|---|---|\n",
    ]
    for k in ("MINOR", "MODERATE", "SEVERE"):
        lines.append(f"| {k.title()} | {sev[k]['count']} | {sev[k]['pct']:.1f}% |\n")
    lines += [
        "\n### Severity heuristic (exact rule)\n\n```\n", SEVERITY_RULE, "```\n",
        "\n## Estimated repairability\n\n",
        "| repairability | count | % of incorrect sentences |\n|---|---|---|\n",
    ]
    for k in ("HIGH", "MEDIUM", "LOW"):
        lines.append(f"| {k.title()} | {rep[k]['count']} | {rep[k]['pct']:.1f}% |\n")
    lines += [
        "\n### Repairability heuristic (exact rule)\n\n```\n", REPAIRABILITY_RULE, "```\n",
        "\n> These are analysis categories only. They are **not** clinically or "
        "scientifically validated, and a `HIGH` rating does **not** mean the sentence "
        "can actually be repaired — only that enough context survives that a repair "
        "model is worth testing. That can only be settled by training and evaluating "
        "a model.\n",
        "\n## Existing train/test split composition\n\n",
        "Reported as-is. No new split was created and nothing was trained.\n\n",
        "| split | total | words | sentences |\n|---|---|---|---|\n",
    ]
    for split in ("train", "test"):
        c = comp.get(split, {})
        lines.append(f"| {split} | {c.get('total', 0)} | {c.get('words', 0)} | "
                     f"{c.get('sentences', 0)} |\n")
    lines.append(
        f"\nThe existing test split holds only **{comp.get('test', {}).get('sentences', 0)} "
        "sentences**, which is far too few to support a reliable sentence-level "
        "conclusion on its own.\n")

    # --- examples ---------------------------------------------------------
    lines.append("\n## Informative sentence errors\n")
    lines.append(
        "\nSelected to span the range, not to flatter the results: the highest-WER "
        "cases in each repairability band are included alongside the cleanest ones.\n")
    for band, blurb in (
        ("HIGH", "Most of the sentence survives; the error is local."),
        ("MEDIUM", "Some context survives, but several words are wrong or ambiguous."),
        ("LOW", "Too much is missing, hallucinated or unrelated for text-only repair."),
    ):
        subset = errors[errors["repairability"] == band]
        lines.append(f"\n### Estimated repairability: {band.title()} "
                     f"({len(subset)} sentences)\n\n_{blurb}_\n")
        if subset.empty:
            lines.append("\n_none_\n")
            continue
        ordered = subset.sort_values("WER")
        picked = pd.concat([ordered.head(3), ordered.tail(3)]).drop_duplicates("sample_id")
        for _, row in picked.iterrows():
            lines.append(example_block(row))

    # --- limitations ------------------------------------------------------
    lines += [
        "\n## Dataset limitations\n\n",
        f"- Only **{metrics['total_samples']} samples** in total "
        f"({comp.get('train', {}).get('total', 0)} train / "
        f"{comp.get('test', {}).get('total', 0)} test).\n",
        f"- The **majority are isolated words** ({w['num_samples']} of "
        f"{metrics['total_samples']}); only **{s['num_samples']}** are "
        "sentences/phrases.\n",
        f"- The existing test split contains only "
        f"{comp.get('test', {}).get('sentences', 0)} sentences — **too small for strong "
        "conclusions**.\n",
        "- **No speaker IDs are available.** The dataset exposes only `audio` and "
        "`text`; the stored audio `path` is null for every row. No speaker IDs were "
        "invented.\n",
        "- **Speaker-disjoint evaluation therefore cannot be performed.** Utterances "
        "from the same speaker are almost certainly on both sides of any split.\n",
        "- The existing split also repeats prompts: the 770 rows contain only 335 "
        "unique ground-truth texts, and 65 texts appear in both train and test.\n",
        "- This is a **male dysarthric TORGO subset**. It is not all of TORGO and "
        "**should not be treated as representative of all people with dysarthria**. "
        "No accessibility or clinical claims are drawn from it.\n",
        "- Results describe one ASR configuration (`base.en`) on this subset only.\n",
    ]
    if metrics["failed_samples"]:
        lines.append("\n## Samples with no ASR output\n\n"
                     "Retained in the dataset, not discarded. See "
                     "`results/failed_samples.csv`.\n\n")
        for f in metrics["failed_samples"]:
            lines.append(f"- `{f['sample_id']}` — ground truth `\"{f['ground_truth']}\"` — "
                         "`vad_filter` returned 0 segments, so the transcript is empty and "
                         "no confidence was reported.\n")

    REPORT_PATH.write_text("".join(lines))


if __name__ == "__main__":
    main()
