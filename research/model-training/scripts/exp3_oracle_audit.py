"""Experiment 3, Stage 1 -- selector oracle audit of the cached hybrid lists.

    python scripts/exp3_oracle_audit.py

Reads only cached artifacts (nbest_cache.jsonl + repair_pairs.csv +
loso_folds.csv). Nothing is transcribed, trained, or modified.

Hybrid list construction is IDENTICAL to System C:
  H1 = cached production 1-best (A0); H2..H5 = ct2 top-4 unique, deduped vs A0.

Label rule (deterministic, training-only; reference never used at inference):
  1. per-hypothesis WER vs reference
  2. min-WER set
  3. H1 in the min set -> KEEP_A0 (explicit conservative bias)
  4. else the min-WER hypothesis with the best decoder rank wins; list order
     already encodes rank, so this is the lowest index. (The edit-distance
     tie-break in the spec never fires because rank order is total; noted.)

Writes results/experiment3_selector/{oracle_analysis.json,label_distribution.json,
oracle_rows.csv}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jiwer
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT = Path(__file__).resolve().parent.parent
LT = PROJECT / "data" / "large_torgo"
OUT = PROJECT / "results" / "experiment3_selector"

TIERS = {"M03": "easy", "F04": "easy", "F03": "medium", "M05": "medium",
         "M02": "hard", "F01": "hard", "M01": "hard", "M04": "hard"}
C_WER = 0.2723  # frozen System C result


def row_wer(ref, hyp):
    return jiwer.wer(ref, hyp if hyp.strip() else "*")


def corpus_wer(refs, hyps):
    if not len(refs):
        return float("nan")
    return jiwer.process_words(list(refs),
                               [h if str(h).strip() else "*" for h in hyps]).wer


def main() -> None:
    pairs = pd.read_csv(LT / "repair_pairs.csv", keep_default_na=False, na_values=[""])
    for c in ("asr_transcript_normalized", "ground_truth_normalized"):
        pairs[c] = pairs[c].fillna("")
    folds = pd.read_csv(LT / "loso_folds.csv")
    unseen_ids = set(folds[(folds["role"] == "test") & (folds["unseen_prompt"])]["sample_id"])

    nbest = {}
    for line in (LT / "nbest" / "nbest_cache.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            nbest[r["sample_id"]] = [h["processed"] for h in r["hypotheses"]]

    dys = pairs[pairs["speaker_group"] == "dysarthric"].copy()
    rows = []
    for _, r in dys.iterrows():
        sid = r["sample_id"]
        if sid not in nbest:
            continue
        a0 = r["asr_transcript_normalized"]
        hyps, seen = ([a0], {a0}) if a0 else ([], set())
        for h in nbest[sid]:
            if h and h not in seen:
                seen.add(h); hyps.append(h)
            if len(hyps) >= 5:
                break
        if not hyps:
            hyps = [""]
        ref = r["ground_truth_normalized"]
        wers = [row_wer(ref, h) for h in hyps]
        best = min(wers)
        tied = [i for i, w in enumerate(wers) if abs(w - best) < 1e-9]
        label_idx = 0 if 0 in tied else tied[0]
        rows.append({
            "sample_id": sid, "speaker_id": r["speaker_id"],
            "tier": TIERS[r["speaker_id"]], "repairability": r["repairability"],
            "unseen_prompt": sid in unseen_ids,
            "ref": ref, "a0": a0, "n_hyps": len(hyps),
            "a0_wer": wers[0], "oracle_wer": best,
            "oracle_idx": label_idx, "n_tied": len(tied),
            "a0_in_tie": 0 in tied,
            "oracle_text": hyps[label_idx],
            "ref_in_list": ref in hyps,
            "label": "KEEP_A0" if label_idx == 0 else f"H{label_idx + 1}",
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "oracle_rows.csv", index=False)
    n = len(df)

    def block(g, name):
        return {
            "subset": name, "n": len(g),
            "a0_wer": corpus_wer(g["ref"], g["a0"]),
            "oracle_wer": corpus_wer(g["ref"], g["oracle_text"]),
            "oracle_gain": corpus_wer(g["ref"], g["a0"]) - corpus_wer(g["ref"], g["oracle_text"]),
            "oracle_exact": float((g["ref"] == g["oracle_text"]).mean()),
            "a0_exact": float((g["ref"] == g["a0"]).mean()),
            "ref_in_list": float(g["ref_in_list"].mean()),
            "a0_is_best": float(g["a0_in_tie"].mean()),
            "better_in_H2_H5": float((~g["a0_in_tie"]).mean()),
        }

    overall = block(df, "all 683 dysarthric sentences")
    a0_correct = df[df["ref"] == df["a0"]]
    result = {
        "hybrid_list": "H1=A0 production 1-best; H2-H5=frozen ct2 top-4 unique",
        "overall": overall,
        "per_speaker": {s: block(g, s) for s, g in df.groupby("speaker_id")},
        "per_tier": {t: block(g, t) for t, g in df.groupby("tier")},
        "unseen_prompts": block(df[df["unseen_prompt"]], "unseen prompts"),
        "low": block(df[df["repairability"] == "LOW"], "LOW"),
        "by_repairability": {k: block(g, k) for k, g in df.groupby("repairability")},
        "a0_already_correct": {
            "n": len(a0_correct),
            "oracle_stays_a0": float(a0_correct["a0_in_tie"].mean()),
            "oracle_wer": corpus_wer(a0_correct["ref"], a0_correct["oracle_text"]),
        },
        "ties": {"mean_tied": float(df["n_tied"].mean()),
                 "multi_tie_rate": float((df["n_tied"] > 1).mean())},
        "mean_list_length": float(df["n_hyps"].mean()),
        "comparison_vs_C": {
            "C_wer": C_WER,
            "selector_oracle_wer": overall["oracle_wer"],
            "oracle_beats_C_by": C_WER - overall["oracle_wer"],
        },
    }
    (OUT / "oracle_analysis.json").write_text(json.dumps(result, indent=2, default=str))

    labels = {
        "overall": df["label"].value_counts().to_dict(),
        "by_speaker": {s: g["label"].value_counts().to_dict() for s, g in df.groupby("speaker_id")},
        "by_tier": {t: g["label"].value_counts().to_dict() for t, g in df.groupby("tier")},
        "by_repairability": {k: g["label"].value_counts().to_dict()
                             for k, g in df.groupby("repairability")},
        "rule": "min WER; H1 ties -> KEEP_A0; else lowest index (=best decoder rank)",
    }
    (OUT / "label_distribution.json").write_text(json.dumps(labels, indent=2))

    print(f"=== SELECTOR ORACLE AUDIT ({n} dysarthric sentences) ===")
    print(f"A0 WER {overall['a0_wer']:.4f} -> hybrid oracle {overall['oracle_wer']:.4f} "
          f"(gain {overall['oracle_gain']:+.4f})")
    print(f"oracle exact {overall['oracle_exact']*100:.1f}% (A0 {overall['a0_exact']*100:.1f}%) | "
          f"ref in list {overall['ref_in_list']*100:.1f}%")
    print(f"A0 already best: {overall['a0_is_best']*100:.1f}% | better in H2-H5: "
          f"{overall['better_in_H2_H5']*100:.1f}%")
    print(f"C = {C_WER} | oracle beats C by {C_WER - overall['oracle_wer']:+.4f}")
    print(f"mean list length {result['mean_list_length']:.2f} | mean tied {result['ties']['mean_tied']:.2f} "
          f"| multi-tie {result['ties']['multi_tie_rate']*100:.1f}%")
    print(f"\nA0-already-correct rows: n={len(a0_correct)}, oracle keeps A0 "
          f"{result['a0_already_correct']['oracle_stays_a0']*100:.1f}%")
    print("\nlabel distribution:", labels["overall"])
    print("\n=== PER SPEAKER (A0 -> oracle) ===")
    for s in sorted(result["per_speaker"], key=lambda x: result["per_speaker"][x]["a0_wer"]):
        b = result["per_speaker"][s]
        print(f"  {s} ({TIERS[s]:<6}) n={b['n']:>3}  {b['a0_wer']:.4f} -> {b['oracle_wer']:.4f} "
              f"(gain {b['oracle_gain']:+.4f})  A0-best {b['a0_is_best']*100:.0f}%")
    for name in ("unseen_prompts", "low"):
        b = result[name]
        print(f"\n{name}: n={b['n']}  {b['a0_wer']:.4f} -> {b['oracle_wer']:.4f} "
              f"(gain {b['oracle_gain']:+.4f})  ref-in-list {b['ref_in_list']*100:.0f}%")
    print("\nby repairability (A0 -> oracle):")
    for k, b in result["by_repairability"].items():
        print(f"  {k:<8} n={b['n']:>3}  {b['a0_wer']:.4f} -> {b['oracle_wer']:.4f}")


if __name__ == "__main__":
    main()
