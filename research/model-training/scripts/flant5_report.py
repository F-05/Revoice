"""Score System C against the pre-registered criteria; report A0/A1/B/C.

    python scripts/flant5_report.py

Writes results/t5_nbest/{evaluation.json,final_report.md,low_rewrites.csv}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jiwer
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import normalize_text  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "results" / "t5_nbest"
LT_DATA = PROJECT / "data" / "large_torgo"

TIERS = {"M03": "easy", "F04": "easy", "F03": "medium", "M05": "medium",
         "M02": "hard", "F01": "hard", "M01": "hard", "M04": "hard"}
SPK_TOL, SPK_IMP = 0.005, 0.010
STRETCH_WER = 0.15


def corpus_wer(refs, hyps):
    if not len(refs):
        return float("nan")
    return jiwer.process_words(list(refs),
                               [h if str(h).strip() else "*" for h in hyps]).wer


def main() -> None:
    test = pd.read_csv(OUT / "loso_predictions.csv", keep_default_na=False, na_values=[""])
    ctl = pd.read_csv(OUT / "control_predictions.csv", keep_default_na=False, na_values=[""])
    for f in (test, ctl):
        for c in ("ground_truth_normalized", "asr_transcript_normalized", "t5_normalized"):
            f[c] = f[c].fillna("")
    test["tier"] = test["speaker_id"].map(TIERS)

    # A1 map from the nbest cache
    a1 = {}
    for line in (LT_DATA / "nbest" / "nbest_cache.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            a1[r["sample_id"]] = r["a1_top1"]
    test["a1"] = test["sample_id"].map(a1).fillna("")

    refs = test["ground_truth_normalized"]
    systems = {
        "A0_production_1best": corpus_wer(refs, test["asr_transcript_normalized"]),
        "A1_ct2_top1_no_repair": corpus_wer(refs, test["a1"]),
        "B_1best_t5small_frozen": 0.3225,  # frozen result, results/t5_loso/
        "C_hybrid_nbest_flant5": corpus_wer(refs, test["t5_normalized"]),
    }

    def block(g):
        n = len(g)
        return {"n": n,
                "asr_wer": corpus_wer(g["ground_truth_normalized"], g["asr_transcript_normalized"]),
                "c_wer": corpus_wer(g["ground_truth_normalized"], g["t5_normalized"]),
                "asr_exact": float((g["ground_truth_normalized"] == g["asr_transcript_normalized"]).mean()),
                "c_exact": float((g["ground_truth_normalized"] == g["t5_normalized"]).mean()),
                "improved": int((g["outcome"] == "IMPROVED").sum()),
                "unchanged": int((g["outcome"] == "UNCHANGED").sum()),
                "worsened": int((g["outcome"] == "WORSENED").sum()),
                "edited": int(g["t5_edited"].sum())}

    agg = block(test)
    per_speaker = {}
    for s, g in test.groupby("speaker_id"):
        b = block(g)
        b["tier"] = TIERS[s]
        b["delta"] = b["c_wer"] - b["asr_wer"]
        per_speaker[s] = b
    tiers = {t: block(g) for t, g in test.groupby("tier")}
    unseen = block(test[test["unseen_prompt"] == True])  # noqa: E712

    low = test[test["repairability"] == "LOW"].copy()
    low_b = block(low)
    low_rewrites = low[low["t5_edited"]].copy()
    low_rewrites[["sample_id", "speaker_id", "ground_truth_normalized",
                  "asr_transcript_normalized", "t5_normalized", "asr_wer_row",
                  "t5_wer_row", "outcome"]].to_csv(OUT / "low_rewrites.csv", index=False)

    correct = test[test["ground_truth_normalized"] == test["asr_transcript_normalized"]]
    preservation = float((~correct["t5_edited"]).mean()) if len(correct) else None

    reparable = test[test["repairability"].isin(["HIGH", "MEDIUM"])]
    rep_b = block(reparable)

    edits = test[test["t5_edited"]]
    edit_stats = {"edit_rate": float(test["t5_edited"].mean()),
                  "edits": int(len(edits)),
                  "edits_improved": int((edits["outcome"] == "IMPROVED").sum()),
                  "edits_worsened": int((edits["outcome"] == "WORSENED").sum()),
                  "edit_precision": (float((edits["outcome"] == "IMPROVED").mean())
                                     if len(edits) else None)}

    ctl_rows = []
    for fold, g in ctl.groupby("fold"):
        cc = g[g["ground_truth_normalized"] == g["asr_transcript_normalized"]]
        ctl_rows.append({"fold": int(fold),
                         "asr_wer": corpus_wer(g["ground_truth_normalized"], g["asr_transcript_normalized"]),
                         "c_wer": corpus_wer(g["ground_truth_normalized"], g["t5_normalized"]),
                         "preservation": float((~cc["t5_edited"]).mean())})
    cf = pd.DataFrame(ctl_rows)
    control = {"asr_wer": float(cf["asr_wer"].iloc[0]),
               "c_wer_mean": float(cf["c_wer"].mean()),
               "c_wer_worst": float(cf["c_wer"].max()),
               "degradation_mean": float(cf["c_wer"].mean() - cf["asr_wer"].iloc[0]),
               "preservation_mean": float(cf["preservation"].mean()),
               "preservation_worst": float(cf["preservation"].min())}

    delta = agg["c_wer"] - agg["asr_wer"]
    degraded = [s for s, b in per_speaker.items() if b["delta"] > SPK_TOL]
    with_err = {s for s, g in test.groupby("speaker_id")
                if g["repairability"].isin(["HIGH", "MEDIUM"]).sum() >= 10}
    clear_imp = [s for s, b in per_speaker.items()
                 if -b["delta"] >= SPK_IMP and s in with_err]
    ratio = agg["improved"] / agg["worsened"] if agg["worsened"] else float("inf")
    rep_rate = rep_b["improved"] / rep_b["n"] if rep_b["n"] else None
    low_rw_rate = float(low["t5_edited"].mean()) if len(low) else 0.0
    low_no_degrade = low_b["c_wer"] <= low_b["asr_wer"] + 1e-9

    criteria = {
        "1_aggregate_wer_minus_0.010": {"delta": delta, "pass": bool(delta <= -0.010)},
        "2_speakers": {"degraded": degraded, "clear_improvement": clear_imp,
                       "pass": bool((8 - len(degraded)) >= 6 and len(clear_imp) >= 2)},
        "3_ratio_3to1": {"improved": agg["improved"], "worsened": agg["worsened"],
                         "ratio": ratio, "pass": bool(ratio >= 3.0)},
        "4_reparable_5pct": {"rate": rep_rate,
                             "pass": bool(rep_rate is not None and rep_rate >= 0.05)},
        "5_preservation_98": {"rate": preservation,
                              "pass": bool(preservation is not None and preservation >= 0.98)},
        "6_control_wer_0.002": {"degradation": control["degradation_mean"],
                                "pass": bool(control["degradation_mean"] <= 0.002)},
        "7_control_preservation_99": {"rate": control["preservation_mean"],
                                      "pass": bool(control["preservation_mean"] >= 0.99)},
        "8_unseen_prompt_not_worse": {"asr": unseen["asr_wer"], "c": unseen["c_wer"],
                                      "pass": bool(unseen["c_wer"] <= unseen["asr_wer"] + 1e-9)},
        "9_low": {"rewrite_rate": low_rw_rate, "no_aggregate_degradation": bool(low_no_degrade),
                  "low_asr_wer": low_b["asr_wer"], "low_c_wer": low_b["c_wer"],
                  "rewrites": int(len(low_rewrites)),
                  "pass": bool(low_rw_rate <= 0.10 and low_no_degrade)},
    }
    overall = all(c["pass"] for c in criteria.values())
    stretch = {"target": STRETCH_WER, "value": agg["c_wer"],
               "met": bool(agg["c_wer"] <= STRETCH_WER)}

    report = {"systems_aggregate_wer": systems,
              "comparisons": {
                  "A0_vs_A1_decoder_effect": systems["A1_ct2_top1_no_repair"] - systems["A0_production_1best"],
                  "A1_vs_C_repair_effect": systems["C_hybrid_nbest_flant5"] - systems["A1_ct2_top1_no_repair"],
                  "A0_vs_C_full_system": systems["C_hybrid_nbest_flant5"] - systems["A0_production_1best"],
                  "A0_vs_B_old_architecture": systems["B_1best_t5small_frozen"] - systems["A0_production_1best"]},
              "aggregate": agg, "per_speaker": per_speaker, "tiers": tiers,
              "unseen_prompt": unseen, "low": criteria["9_low"],
              "preservation": {"n_correct": int(len(correct)), "rate": preservation},
              "edit_stats": edit_stats, "control": control,
              "criteria": criteria, "all_pass": overall, "stretch_goal": stretch}
    (OUT / "evaluation.json").write_text(json.dumps(report, indent=2, default=str))

    print("=== SYSTEMS (aggregate WER, 683 held-out sentences) ===")
    for k, v in systems.items():
        print(f"  {k}: {v:.4f}")
    print("\n=== COMPARISONS (negative = second system better) ===")
    for k, v in report["comparisons"].items():
        print(f"  {k}: {v:+.4f}")
    print(f"\nC aggregate: WER {agg['asr_wer']:.4f} -> {agg['c_wer']:.4f} ({delta:+.4f}) | "
          f"exact {agg['asr_exact'] * 100:.1f}% -> {agg['c_exact'] * 100:.1f}%")
    print(f"improved {agg['improved']} / unchanged {agg['unchanged']} / worsened {agg['worsened']}"
          f" | ratio {ratio:.2f}")
    print(f"edit rate {edit_stats['edit_rate']:.3f} | precision {edit_stats['edit_precision']}")
    print(f"preservation {preservation}")
    print("\n=== PER SPEAKER ===")
    for s in sorted(per_speaker, key=lambda x: per_speaker[x]["asr_wer"]):
        b = per_speaker[s]
        print(f"  {s} ({b['tier']:<6}) n={b['n']:>3} ASR {b['asr_wer']:.4f} -> C {b['c_wer']:.4f} "
              f"({b['delta']:+.4f}) +{b['improved']}/-{b['worsened']} edits={b['edited']}")
    print(f"\nunseen prompts n={unseen['n']}: ASR {unseen['asr_wer']:.4f} -> C {unseen['c_wer']:.4f} "
          f"+{unseen['improved']}/-{unseen['worsened']}")
    print(f"LOW: n={low_b['n']} rewrites={len(low_rewrites)} rate={low_rw_rate:.3f} "
          f"WER {low_b['asr_wer']:.4f} -> {low_b['c_wer']:.4f}")
    if len(low_rewrites):
        print(low_rewrites["outcome"].value_counts().to_string())
    print(f"control: {control['asr_wer']:.4f} -> mean {control['c_wer_mean']:.4f} "
          f"(worst {control['c_wer_worst']:.4f}) preservation {control['preservation_mean']:.4f}")
    print("\n=== CRITERIA ===")
    for k, c in criteria.items():
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {k}")
    print(f"\nStretch goal WER<=0.15: {'MET' if stretch['met'] else 'NOT MET'} ({agg['c_wer']:.4f})")
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")


if __name__ == "__main__":
    main()
