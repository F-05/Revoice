"""D3 Stage 1 -- failure feature audit of D2's frozen switches. NO TRAINING.

    python scripts/exp3_d3_audit.py

Ground truth is used ONLY to classify each historical D2 switch as
beneficial/harmful after the fact; every audited feature is computable at
inference time from the hypothesis list alone.

Writes results/experiment3_selector/{d3_audit_features.csv,d3_audit.json,
d3_f03_audit.md}.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_d1 import overlap_f1, word_edit  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
LT = PROJECT / "data" / "large_torgo"
OUT = PROJECT / "results" / "experiment3_selector"


def auc(pos, neg):
    """Mann-Whitney ROC-AUC: P(feature(harmful) > feature(beneficial))."""
    pos, neg = [x for x in pos if x == x], [x for x in neg if x == x]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def build_list(sid, a0, nbest):
    cands, seen = ([{"text": a0, "score": None}], {a0}) if a0 else ([], set())
    for h in nbest[sid]:
        t = h["processed"]
        if t and t not in seen:
            seen.add(t)
            cands.append({"text": t, "score": h["score"]})
        if len(cands) >= 5:
            break
    return cands


def features_for(cand_idx, cands, a0):
    """The 20 audited features for one candidate within its list."""
    texts = [c["text"] for c in cands]
    scores = [c["score"] for c in cands]
    cand = texts[cand_idx]
    cw, aw = cand.split(), a0.split()
    cset, aset = set(cw), set(aw)
    others = [t for i, t in enumerate(texts) if i != cand_idx]
    other_sets = [set(t.split()) for t in others]
    non_a0_others = [set(t.split()) for i, t in enumerate(texts)
                     if i != cand_idx and t != a0]
    support = Counter(w for t in texts for w in set(t.split()))

    changed = [w for w in cw if w not in aset]
    deleted = [w for w in aw if w not in cset]
    ed = word_edit(cw, aw)
    cand_f1s = [overlap_f1(cset, o) for o in other_sets] or [1.0]
    a0_f1s = [overlap_f1(aset, set(t.split())) for t in texts if t != a0] or [1.0]
    ct2 = [s for s in scores if s is not None]
    spread = (max(ct2) - min(ct2)) if len(ct2) > 1 else 0.0
    cand_score = scores[cand_idx]

    changed_attested = [w for w in changed if support[w] >= 2]  # >=1 other hyp
    return {
        "f01_edit_to_a0": ed,
        "f02_edit_norm": ed / max(len(cw), len(aw), 1),
        "f03_f1_with_a0": overlap_f1(cset, aset),
        "f04_mean_f1_others": float(np.mean(cand_f1s)),
        "f05_a0_mean_consensus": float(np.mean(a0_f1s)),
        "f06_consensus_advantage": float(np.mean(cand_f1s)) - float(np.mean(a0_f1s)),
        "f07_min_overlap_other": float(min(cand_f1s)),
        "f08_max_overlap_other": float(max(cand_f1s)),
        "f09_n_hyps_supporting_changed": (
            sum(any(w in o for w in changed) for o in non_a0_others)
            if changed else len(non_a0_others)),
        "f10_changed_attested_frac": (len(changed_attested) / len(changed)
                                      if changed else 1.0),
        "f11_changed_unique_frac": (sum(support[w] == 1 for w in changed) / len(changed)
                                    if changed else 0.0),
        "f12_disputed_support": (float(np.mean([support[w] >= 2 for w in changed]))
                                 if changed else 1.0),
        "f13_n_agree_disputed": (int(np.median([sum(w in o for o in non_a0_others)
                                                for w in changed])) if changed else 0),
        "f14_list_disagreement": 1.0 - float(np.mean(
            [overlap_f1(set(a.split()), set(b.split()))
             for i, a in enumerate(texts) for b in texts[i + 1:]] or [1.0])),
        "f15_margin_score_norm": ((cand_score - min(ct2)) / spread
                                  if cand_score is not None and spread > 0 else 0.0),
        "f16_rank": cand_idx,
        "f17_len_diff_a0": len(cw) - len(aw),
        "f18_introduces_novel_words": float(any(support[w] == 1 for w in changed)),
        "f19_deletes_supported_words": float(any(support[w] >= 3 for w in deleted)),
        "f20_lexical_outlier": float(np.mean(cand_f1s) < np.mean(a0_f1s) - 0.1),
    }


def main() -> None:
    d2 = pd.read_csv(OUT / "d2_predictions.csv", keep_default_na=False, na_values=[""])
    for c in ("ref", "a0", "d2_text"):
        d2[c] = d2[c].fillna("")
    nbest = {}
    for line in (LT / "nbest" / "nbest_cache.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            nbest[r["sample_id"]] = r["hypotheses"]

    switches = d2[d2["switched"]].copy()
    rows = []
    for _, s in switches.iterrows():
        cands = build_list(s["sample_id"], s["a0"], nbest)
        texts = [c["text"] for c in cands]
        idx = texts.index(s["d2_text"])
        f = features_for(idx, cands, s["a0"])
        rows.append({"sample_id": s["sample_id"], "speaker_id": s["speaker_id"],
                     "outcome": s["outcome"], "margin": s["margin"],
                     "ref": s["ref"], "a0": s["a0"], "picked": s["d2_text"],
                     "all_hyps": " || ".join(texts), **f})
    feats = pd.DataFrame(rows)
    feats.to_csv(OUT / "d3_audit_features.csv", index=False)

    ben = feats[feats["outcome"] == "IMPROVED"]
    harm = feats[feats["outcome"] == "WORSENED"]
    neutral = feats[feats["outcome"] == "UNCHANGED"]
    print(f"D2 switches: {len(feats)} = {len(ben)} beneficial / "
          f"{len(harm)} harmful / {len(neutral)} neutral")

    feature_cols = [c for c in feats.columns if c.startswith("f")]
    summary = {}
    print(f"\n{'feature':<32} {'benef mean':>10} {'harm mean':>10} {'AUC(harm>ben)':>13}")
    for c in feature_cols:
        a = auc(list(harm[c]), list(ben[c]))
        summary[c] = {
            "beneficial": {"mean": float(ben[c].mean()), "median": float(ben[c].median()),
                           "std": float(ben[c].std()), "q25": float(ben[c].quantile(.25)),
                           "q75": float(ben[c].quantile(.75))},
            "harmful": {"mean": float(harm[c].mean()), "median": float(harm[c].median()),
                        "std": float(harm[c].std()), "q25": float(harm[c].quantile(.25)),
                        "q75": float(harm[c].quantile(.75))},
            "auc_harm_gt_ben": a,
        }
        print(f"{c:<32} {ben[c].mean():>10.3f} {harm[c].mean():>10.3f} "
              f"{a if a is None else round(a, 3):>13}")
    summary["margin"] = {"beneficial_mean": float(ben["margin"].mean()),
                         "harmful_mean": float(harm["margin"].mean()),
                         "auc": auc(list(harm["margin"]), list(ben["margin"]))}
    print(f"\nD2 margin itself: benef {ben['margin'].mean():.3f} vs harm "
          f"{harm['margin'].mean():.3f}, AUC {summary['margin']['auc']:.3f}")
    (OUT / "d3_audit.json").write_text(json.dumps(summary, indent=2))

    # ---- F03 deep dive ----------------------------------------------------
    lines = ["# F03 harmful-switch audit (D2, frozen)\n"]
    f03_harm = feats[(feats["speaker_id"] == "F03") & (feats["outcome"] == "WORSENED")]
    key_feats = ["f04_mean_f1_others", "f06_consensus_advantage", "f10_changed_attested_frac",
                 "f11_changed_unique_frac", "f18_introduces_novel_words", "f20_lexical_outlier"]
    for _, r in f03_harm.iterrows():
        lines.append(f"\n## `{r['sample_id']}` (margin {r['margin']:.3f})\n\n```")
        lines.append(f"\nreference: {r['ref']}\nA0:        {r['a0']}\npicked:    {r['picked']}")
        lines.append(f"\nlist:      {r['all_hyps']}\n```\n")
        lines.append("| feature | value |\n|---|---|\n")
        for k in key_feats:
            lines.append(f"| {k} | {r[k]:.3f} |\n")
    # matched beneficial switches at similar margins
    if len(f03_harm):
        lo, hi = f03_harm["margin"].min() - 0.05, f03_harm["margin"].max() + 0.05
        matched = ben[(ben["margin"] >= lo) & (ben["margin"] <= hi)]
        lines.append(f"\n# Matched beneficial switches (margin in [{lo:.2f}, {hi:.2f}], "
                     f"n={len(matched)})\n")
        for _, r in matched.head(8).iterrows():
            lines.append(f"\n## `{r['sample_id']}` {r['speaker_id']} "
                         f"(margin {r['margin']:.3f})\n\n```")
            lines.append(f"\nreference: {r['ref']}\nA0:        {r['a0']}\npicked:    {r['picked']}\n```\n")
            lines.append("| feature | value |\n|---|---|\n")
            for k in key_feats:
                lines.append(f"| {k} | {r[k]:.3f} |\n")
        for k in key_feats:
            lines.append(f"\n{k}: F03-harm mean {f03_harm[k].mean():.3f} vs "
                         f"matched-benef mean {matched[k].mean():.3f}")
    (OUT / "d3_f03_audit.md").write_text("".join(lines))
    print(f"\nWrote d3_audit_features.csv, d3_audit.json, d3_f03_audit.md")


if __name__ == "__main__":
    main()
