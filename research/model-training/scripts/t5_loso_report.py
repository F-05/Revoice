"""Score the LOSO run against the pre-registered success criteria.

    python scripts/t5_loso_report.py

Reads results/t5_loso/{loso_predictions.csv,control_predictions.csv,
loso_results.json} and writes evaluation.json + loso_report.md with an explicit
pass/fail per criterion. The criteria were fixed before training (see
results/large_torgo/experiment2_design.md plus Nathan's amendments).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jiwer
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "results" / "t5_loso"

# "No meaningful degradation" for a speaker = WER rise of at most this.
SPEAKER_DEGRADATION_TOL = 0.005
# "Clear improvement" for a speaker = WER drop of at least this.
SPEAKER_IMPROVEMENT_MIN = 0.010


def corpus_wer(refs, hyps):
    if not len(refs):
        return float("nan")
    return jiwer.process_words(list(refs),
                               [h if str(h).strip() else "*" for h in hyps]).wer


def block(frame):
    n = len(frame)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "asr_wer": corpus_wer(frame["ground_truth_normalized"],
                              frame["asr_transcript_normalized"]),
        "t5_wer": corpus_wer(frame["ground_truth_normalized"], frame["t5_normalized"]),
        "asr_exact": float((frame["ground_truth_normalized"]
                            == frame["asr_transcript_normalized"]).mean()),
        "t5_exact": float((frame["ground_truth_normalized"]
                           == frame["t5_normalized"]).mean()),
        "improved": int((frame["outcome"] == "IMPROVED").sum()),
        "unchanged": int((frame["outcome"] == "UNCHANGED").sum()),
        "worsened": int((frame["outcome"] == "WORSENED").sum()),
        "edited": int(frame["t5_edited"].sum()),
    }


def main() -> None:
    test = pd.read_csv(OUT / "loso_predictions.csv", keep_default_na=False, na_values=[""])
    ctl = pd.read_csv(OUT / "control_predictions.csv", keep_default_na=False, na_values=[""])
    for frame in (test, ctl):
        for c in ("ground_truth_normalized", "asr_transcript_normalized", "t5_normalized"):
            frame[c] = frame[c].fillna("")
    meta = json.loads((OUT / "loso_results.json").read_text())

    agg = block(test)
    per_speaker = {}
    for speaker, g in test.groupby("speaker_id"):
        b = block(g)
        b["tier"] = g["tier"].iloc[0]
        b["delta_wer"] = b["t5_wer"] - b["asr_wer"]
        per_speaker[speaker] = b
    tiers = {t: block(g) for t, g in test.groupby("tier")}
    unseen = block(test[test["unseen_prompt"] == True])  # noqa: E712

    low = test[test["repairability"] == "LOW"]
    low_block = block(low)
    low_block["rewritten"] = int(low["t5_edited"].sum())
    low_block["rewritten_worsened"] = int(
        ((low["t5_edited"]) & (low["outcome"] == "WORSENED")).sum())

    # Correct-input preservation on test.
    correct = test[test["ground_truth_normalized"] == test["asr_transcript_normalized"]]
    preservation = {
        "correct_asr_examples": int(len(correct)),
        "preserved": int((~correct["t5_edited"]).sum()),
        "modified": int(correct["t5_edited"].sum()),
        "rate": float((~correct["t5_edited"]).mean()) if len(correct) else None,
    }

    # Erroneous + trainable-category rows: the population repair is FOR.
    reparable = test[test["repairability"].isin(["HIGH", "MEDIUM"])]
    reparable_block = block(reparable)
    reparable_improved_rate = (reparable_block["improved"] / reparable_block["n"]
                               if reparable_block["n"] else None)

    # Edit rate / precision over all test rows.
    edits = test[test["t5_edited"]]
    edit_stats = {
        "edit_rate": float(test["t5_edited"].mean()),
        "edits": int(len(edits)),
        "edits_improved": int((edits["outcome"] == "IMPROVED").sum()),
        "edits_worsened": int((edits["outcome"] == "WORSENED").sum()),
        "edit_precision": (float((edits["outcome"] == "IMPROVED").mean())
                           if len(edits) else None),
    }

    # Control: per-fold means (each fold's model sees the same 194 sentences).
    ctl_folds = []
    for fold, g in ctl.groupby("fold"):
        cb = block(g) if "outcome" in g else {}
        correct_c = g[g["ground_truth_normalized"] == g["asr_transcript_normalized"]]
        ctl_folds.append({
            "fold": int(fold),
            "asr_wer": corpus_wer(g["ground_truth_normalized"],
                                  g["asr_transcript_normalized"]),
            "t5_wer": corpus_wer(g["ground_truth_normalized"], g["t5_normalized"]),
            "preservation": float((~correct_c["t5_edited"]).mean()),
        })
    ctl_frame = pd.DataFrame(ctl_folds)
    control_summary = {
        "asr_wer": float(ctl_frame["asr_wer"].iloc[0]),
        "t5_wer_mean": float(ctl_frame["t5_wer"].mean()),
        "t5_wer_worst": float(ctl_frame["t5_wer"].max()),
        "degradation_mean": float(ctl_frame["t5_wer"].mean() - ctl_frame["asr_wer"].iloc[0]),
        "degradation_worst": float(ctl_frame["t5_wer"].max() - ctl_frame["asr_wer"].iloc[0]),
        "preservation_mean": float(ctl_frame["preservation"].mean()),
        "preservation_worst": float(ctl_frame["preservation"].min()),
        "per_fold": ctl_folds,
    }

    # ----- pre-registered criteria ----------------------------------------
    delta = agg["t5_wer"] - agg["asr_wer"]
    degraded = {s: b for s, b in per_speaker.items()
                if b["delta_wer"] > SPEAKER_DEGRADATION_TOL}
    improved_speakers = {s: b for s, b in per_speaker.items()
                         if -b["delta_wer"] >= SPEAKER_IMPROVEMENT_MIN}
    # Speakers that actually contain repairable errors: >=10 HIGH/MEDIUM rows.
    with_errors = {s for s, g in test.groupby("speaker_id")
                   if (g["repairability"].isin(["HIGH", "MEDIUM"])).sum() >= 10}
    improved_with_errors = {s for s in improved_speakers if s in with_errors}

    ratio = (agg["improved"] / agg["worsened"]) if agg["worsened"] else float("inf")
    criteria = {
        "1_aggregate_wer_improves_by_0.010": {
            "value": delta, "threshold": -0.010, "pass": bool(delta <= -0.010)},
        "2_no_meaningful_degradation_6_of_8": {
            "degraded_speakers": sorted(degraded),
            "ok_speakers": 8 - len(degraded),
            "clear_improvement_speakers_with_errors": sorted(improved_with_errors),
            "pass": bool((8 - len(degraded)) >= 6 and len(improved_with_errors) >= 2)},
        "3_improved_worsened_ratio_3to1": {
            "improved": agg["improved"], "worsened": agg["worsened"],
            "ratio": ratio, "pass": bool(ratio >= 3.0)},
        "4_reparable_improved_5pct": {
            "reparable_rows": reparable_block["n"],
            "improved": reparable_block["improved"],
            "rate": reparable_improved_rate,
            "pass": bool(reparable_improved_rate is not None
                         and reparable_improved_rate >= 0.05)},
        "5_preservation_98pct": {
            "rate": preservation["rate"],
            "pass": bool(preservation["rate"] is not None
                         and preservation["rate"] >= 0.98)},
        "6_control_safe": {
            "degradation_mean": control_summary["degradation_mean"],
            "preservation_mean": control_summary["preservation_mean"],
            "pass": bool(control_summary["degradation_mean"] <= 0.002
                         and control_summary["preservation_mean"] >= 0.99)},
        "7_edit_rate_and_precision_reported": {
            **edit_stats, "pass": True},
        "9_low_rewrites_counted_as_failure": {
            "low_rows": low_block["n"], "rewritten": low_block["rewritten"],
            "note": "every rewrite of a LOW row is a failure by definition",
            "pass": bool(low_block["rewritten"] == 0)},
    }
    overall = all(c["pass"] for c in criteria.values())

    report = {
        "aggregate": agg, "per_speaker": per_speaker, "tiers": tiers,
        "unseen_prompt": unseen, "low_inputs": low_block,
        "preservation": preservation, "reparable": reparable_block,
        "edit_stats": edit_stats, "control": control_summary,
        "criteria": criteria, "all_criteria_pass": overall,
        "fold_meta": meta["folds"],
        "definitions": {
            "speaker_degradation_tolerance": SPEAKER_DEGRADATION_TOL,
            "speaker_clear_improvement_min": SPEAKER_IMPROVEMENT_MIN,
            "speakers_with_repairable_errors": ">=10 HIGH/MEDIUM test rows",
        },
    }
    (OUT / "evaluation.json").write_text(json.dumps(report, indent=2, default=str))

    # ----- console + markdown ---------------------------------------------
    print("=== AGGREGATE LOSO (683 sentences, every speaker unseen) ===")
    print(f"  ASR alone: WER {agg['asr_wer']:.4f}  exact {agg['asr_exact'] * 100:.2f}%")
    print(f"  ASR + T5 : WER {agg['t5_wer']:.4f}  exact {agg['t5_exact'] * 100:.2f}%")
    print(f"  delta: {delta:+.4f}")
    print(f"  improved {agg['improved']} | unchanged {agg['unchanged']} | "
          f"worsened {agg['worsened']}")
    print("\n=== PER SPEAKER ===")
    for s in sorted(per_speaker, key=lambda x: per_speaker[x]["asr_wer"]):
        b = per_speaker[s]
        print(f"  {s} ({b['tier']:<6}) n={b['n']:>3}  ASR {b['asr_wer']:.4f} -> "
              f"T5 {b['t5_wer']:.4f} ({b['delta_wer']:+.4f})  "
              f"+{b['improved']}/-{b['worsened']} edits={b['edited']}")
    print("\n=== TIERS ===")
    for t, b in tiers.items():
        print(f"  {t:<6} n={b['n']:>3}  ASR {b['asr_wer']:.4f} -> T5 {b['t5_wer']:.4f}")
    print(f"\nunseen prompts: n={unseen['n']}  ASR {unseen['asr_wer']:.4f} -> "
          f"T5 {unseen['t5_wer']:.4f}  +{unseen['improved']}/-{unseen['worsened']}")
    print(f"LOW inputs: n={low_block['n']}  rewritten={low_block['rewritten']}")
    print(f"preservation: {preservation['preserved']}/{preservation['correct_asr_examples']}"
          f" = {preservation['rate']:.4f}" if preservation["rate"] is not None else "")
    print(f"edit rate {edit_stats['edit_rate']:.3f} | precision "
          f"{edit_stats['edit_precision']}")
    print(f"control: degradation mean {control_summary['degradation_mean']:+.4f} "
          f"worst {control_summary['degradation_worst']:+.4f} | "
          f"preservation mean {control_summary['preservation_mean']:.4f}")
    print("\n=== PRE-REGISTERED CRITERIA ===")
    for name, c in criteria.items():
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {name}")
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")

    lines = ["# Experiment 2 — LOSO results\n",
             f"\nOverall verdict against pre-registered criteria: "
             f"**{'PASS' if overall else 'FAIL'}**\n",
             "\n| criterion | result | pass |\n|---|---|---|\n"]
    for name, c in criteria.items():
        detail = {k: (round(v, 4) if isinstance(v, float) else v)
                  for k, v in c.items() if k != "pass"}
        lines.append(f"| {name} | {detail} | {'✅' if c['pass'] else '❌'} |\n")
    lines.append("\nFull numbers in `evaluation.json`; per-row outputs in "
                 "`loso_predictions.csv` / `control_predictions.csv`.\n")
    (OUT / "loso_report.md").write_text("".join(lines))
    print(f"\nWrote {OUT / 'evaluation.json'} and {OUT / 'loso_report.md'}")


if __name__ == "__main__":
    main()
