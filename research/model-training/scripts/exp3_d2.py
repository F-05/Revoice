"""Experiment 3, Stages 4-5 -- D2: conservative selector with abstention.

    python scripts/exp3_d2.py

FROZEN BEFORE ANY TEST EVALUATION:
  Threshold grid: tau in {0.00, 0.05, 0.10, ..., 0.90} (19 values).
  Per-fold selection uses ONLY that fold's validation speaker, priority order:
    1. validation correct-input preservation >= 99%
       (if no tau achieves it: keep only the taus with maximal preservation)
    2. if any candidate tau yields >= 5 validation switches AND
       improved:worsened >= 4, restrict to taus meeting the ratio; if none
       do (too few edits to be meaningful), skip this step
    3. MINIMIZE validation WER
    4. ties (within 1e-9): choose the LARGEST tau (most conservative)

Inference: margin = P(best non-A0 candidate) - P(A0).
  margin >= tau -> switch (output that candidate)
  else          -> output A0; flagged UNCERTAIN when argmax was non-A0
Text output is always one of H1-H5 (verified; unsupported generation must be 0).
Ground truth is never read at inference. D1 weights are reproduced with the
identical deterministic training procedure and seed.

Writes results/experiment3_selector/{d2_predictions.csv,d2_control_predictions.csv,
d2_evaluation.json,risk_coverage.csv}.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_d1 import FEATURES, build_rows, overlap_f1, train_fold, word_edit  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
LT = PROJECT / "data" / "large_torgo"
OUT = PROJECT / "results" / "experiment3_selector"

TAUS = [round(0.05 * i, 2) for i in range(19)]  # 0.00 .. 0.90
TIERS = {"M03": "easy", "F04": "easy", "F03": "medium", "M05": "medium",
         "M02": "hard", "F01": "hard", "M01": "hard", "M04": "hard"}


def row_wer(ref, hyp):
    return jiwer.wer(ref, hyp if hyp.strip() else "*")


def corpus_wer(refs, hyps):
    if not len(refs):
        return float("nan")
    return jiwer.process_words(list(refs),
                               [h if str(h).strip() else "*" for h in hyps]).wer


def build_control_rows():
    """Feature rows for the 194 control sentences (inference only, no labels)."""
    pairs = pd.read_csv(LT / "repair_pairs.csv", keep_default_na=False, na_values=[""])
    for c in ("asr_transcript_normalized", "ground_truth_normalized"):
        pairs[c] = pairs[c].fillna("")
    ctl = pairs[pairs["split"] == "control_test"]
    nbest = {}
    for line in (LT / "nbest" / "nbest_cache.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            nbest[r["sample_id"]] = r["hypotheses"]

    utts = []
    for _, m in ctl.iterrows():
        sid = m["sample_id"]
        if sid not in nbest:
            continue
        a0 = m["asr_transcript_normalized"]
        conf = m["asr_confidence"] if pd.notna(m["asr_confidence"]) else None
        cands, seen = ([{"text": a0, "is_a0": 1, "ct2_score": None}], {a0}) if a0 else ([], set())
        for h in nbest[sid]:
            t = h["processed"]
            if t and t not in seen:
                seen.add(t)
                cands.append({"text": t, "is_a0": 0, "ct2_score": h["score"]})
            if len(cands) >= 5:
                break
        if not cands:
            cands = [{"text": "", "is_a0": 1, "ct2_score": None}]

        ct2 = [c["ct2_score"] for c in cands if c["ct2_score"] is not None]
        best_ct2 = max(ct2) if ct2 else 0.0
        sorted_ct2 = sorted(ct2, reverse=True)
        lengths = [len(c["text"].split()) for c in cands]
        median_len = float(np.median([l for l in lengths if l > 0]) or 1.0)
        wsets = [set(c["text"].split()) for c in cands]
        a0_words, a0_set = a0.split(), set(a0.split())
        support = Counter(w for ws in wsets for w in ws)

        feats = []
        for i, c in enumerate(cands):
            words = c["text"].split()
            others = [wsets[j] for j in range(len(cands)) if j != i]
            cons = float(np.mean([overlap_f1(wsets[i], o) for o in others])) if others else 1.0
            sup2 = (np.mean([support[w] >= 3 for w in words]) if words else 0.0)
            disputed = [w for w in words if w not in a0_set]
            dsup = float(np.mean([support[w] >= 2 for w in disputed])) if disputed else 1.0
            ed = word_edit(words, a0_words)
            score = c["ct2_score"]
            if score is not None and len(sorted_ct2) > 1:
                pos = sorted_ct2.index(score)
                gap = sorted_ct2[pos] - sorted_ct2[pos + 1] if pos + 1 < len(sorted_ct2) else 0.0
            else:
                gap = 0.0
            feats.append([
                float(c["is_a0"]), float(i),
                float(conf) if (c["is_a0"] and conf is not None) else 0.0,
                float(c["is_a0"] and conf is None),
                float(score) if score is not None else 0.0,
                float(score - best_ct2) if score is not None else 0.0,
                float(gap),
                float(score) / max(len(words), 1) if score is not None else 0.0,
                float(score is None),
                float(ed), float(ed) / max(len(words), len(a0_words), 1),
                float(len(words)), float(len(words)) / median_len if median_len else 0.0,
                cons, float(sup2), dsup,
            ])
        utts.append({"sample_id": sid, "ref": m["ground_truth_normalized"],
                     "a0": a0, "texts": [c["text"] for c in cands],
                     "features": feats})
    return utts


def probabilities(utts, w, mu, sd):
    out = []
    for u in utts:
        x = torch.tensor((np.array(u["features"]) - mu) / sd, dtype=torch.float32)
        with torch.no_grad():
            p = torch.softmax(x @ w, dim=0).numpy()
        a0_idx = next((i for i, t in enumerate(u["texts"]) if t == u["a0"]), None)
        p_a0 = float(p[a0_idx]) if a0_idx is not None else 0.0
        alt = [(float(p[i]), i) for i in range(len(u["texts"])) if i != a0_idx]
        p_alt, alt_idx = max(alt) if alt else (0.0, None)
        out.append({"u": u, "p_a0": p_a0, "p_alt": p_alt, "alt_idx": alt_idx,
                    "margin": p_alt - p_a0, "argmax_is_a0": p_a0 >= p_alt})
    return out


def apply_tau(scored, tau):
    rows = []
    for s in scored:
        u = s["u"]
        switch = s["alt_idx"] is not None and s["margin"] >= tau - 1e-12
        text = u["texts"][s["alt_idx"]] if switch else u["a0"]
        rows.append({**{k: u.get(k) for k in ("sample_id", "speaker_id", "tier",
                                              "repairability", "unseen_prompt",
                                              "ref", "a0")},
                     "d2_text": text, "switched": switch,
                     "uncertain": (not switch) and (not s["argmax_is_a0"]),
                     "margin": s["margin"], "p_a0": s["p_a0"], "p_alt": s["p_alt"]})
    df = pd.DataFrame(rows)
    df["a0_wer_row"] = [row_wer(r, h) for r, h in zip(df["ref"], df["a0"])]
    df["d2_wer_row"] = [row_wer(r, h) for r, h in zip(df["ref"], df["d2_text"])]
    df["outcome"] = ["IMPROVED" if a < b - 1e-9 else "WORSENED" if a > b + 1e-9
                     else "UNCHANGED"
                     for b, a in zip(df["a0_wer_row"], df["d2_wer_row"])]
    return df


def metrics(df):
    n = len(df)
    switches = df[df["switched"]]
    correct = df[df["ref"] == df["a0"]]
    imp, wor = int((df["outcome"] == "IMPROVED").sum()), int((df["outcome"] == "WORSENED").sum())
    low = df[df["repairability"] == "LOW"] if "repairability" in df else df.iloc[0:0]
    up = df[df["unseen_prompt"] == True] if "unseen_prompt" in df else df.iloc[0:0]  # noqa: E712
    return {
        "n": n, "wer": corpus_wer(df["ref"], df["d2_text"]),
        "a0_wer": corpus_wer(df["ref"], df["a0"]),
        "exact": float((df["ref"] == df["d2_text"]).mean()),
        "switch_rate": float(df["switched"].mean()),
        "keep_a0_rate": 1.0 - float(df["switched"].mean()),
        "uncertain_rate": float(df["uncertain"].mean()),
        "improved": imp, "worsened": wor,
        "unchanged": int((df["outcome"] == "UNCHANGED").sum()),
        "ratio": imp / wor if wor else float("inf"),
        "edit_precision": (float((switches["outcome"] == "IMPROVED").mean())
                           if len(switches) else None),
        "preservation": (float((~correct["switched"]).mean()) if len(correct) else None),
        "low_wer": corpus_wer(low["ref"], low["d2_text"]) if len(low) else None,
        "low_a0_wer": corpus_wer(low["ref"], low["a0"]) if len(low) else None,
        "low_harmful_switches": int((low["outcome"] == "WORSENED").sum()) if len(low) else 0,
        "low_switch_rate": float(low["switched"].mean()) if len(low) else None,
        "unseen_wer": corpus_wer(up["ref"], up["d2_text"]) if len(up) else None,
        "unseen_a0_wer": corpus_wer(up["ref"], up["a0"]) if len(up) else None,
        "unseen_improved": int((up["outcome"] == "IMPROVED").sum()) if len(up) else 0,
        "unseen_worsened": int((up["outcome"] == "WORSENED").sum()) if len(up) else 0,
    }


def select_tau(val_scored):
    stats = []
    for tau in TAUS:
        df = apply_tau(val_scored, tau)
        m = metrics(df)
        m["tau"] = tau
        m["n_switches"] = int(df["switched"].sum())
        stats.append(m)
    ok = [m for m in stats if (m["preservation"] is None or m["preservation"] >= 0.99)]
    if not ok:
        best_pres = max(m["preservation"] for m in stats)
        ok = [m for m in stats if m["preservation"] == best_pres]
    ratio_ok = [m for m in ok if m["n_switches"] >= 5 and m["ratio"] >= 4.0]
    pool = ratio_ok if ratio_ok else ok
    best_wer = min(m["wer"] for m in pool)
    tied = [m for m in pool if m["wer"] <= best_wer + 1e-9]
    chosen = max(tied, key=lambda m: m["tau"])  # most conservative
    return chosen["tau"], stats


def main() -> None:
    utts = build_rows()
    control = build_control_rows()
    folds = pd.read_csv(LT / "loso_folds.csv")
    by_id = {u["sample_id"]: u for u in utts}
    print(f"{len(utts)} dysarthric + {len(control)} control utterances")

    all_test, all_ctl, fold_report, rc_rows = [], [], [], []
    for fold in sorted(folds["fold"].unique()):
        fr = folds[folds["fold"] == fold]
        get = lambda role: [by_id[s] for s in fr[fr["role"] == role]["sample_id"]
                            if s in by_id]
        train_u, val_u, test_u = get("train"), get("validation"), get("test")
        w, mu, sd = train_fold(train_u, val_u)

        tau, val_stats = select_tau(probabilities(val_u, w, mu, sd))
        test_scored = probabilities(test_u, w, mu, sd)
        test_df = apply_tau(test_scored, tau)
        test_df["fold"] = fold
        all_test.append(test_df)

        ctl_scored = probabilities(control, w, mu, sd)
        ctl_df = apply_tau(ctl_scored, tau)
        ctl_df["fold"] = fold
        all_ctl.append(ctl_df)

        # risk-coverage sweep on the test fold (REPORTING ONLY, not selection)
        for t in TAUS:
            m = metrics(apply_tau(test_scored, t))
            m.update({"fold": fold, "tau": t,
                      "speaker": test_u[0]["speaker_id"]})
            rc_rows.append(m)

        fold_report.append({"fold": int(fold), "test_speaker": test_u[0]["speaker_id"],
                            "selected_tau": tau,
                            "validation_speaker": val_u[0]["speaker_id"]})
        print(f"fold {fold} ({test_u[0]['speaker_id']}): tau={tau}")

    test = pd.concat(all_test, ignore_index=True)
    ctl = pd.concat(all_ctl, ignore_index=True)
    test.to_csv(OUT / "d2_predictions.csv", index=False)
    ctl.to_csv(OUT / "d2_control_predictions.csv", index=False)

    # aggregate risk-coverage: recompute over pooled test rows per tau
    pooled = []
    for t in TAUS:
        sub = pd.concat([pd.DataFrame(r, index=[0]) for r in rc_rows
                         if r["tau"] == t], ignore_index=True)
        pooled.append({"tau": t, "scope": "per-fold-rows", **{
            k: float(sub[k].mean()) if sub[k].dtype != object else None
            for k in ("wer", "switch_rate", "preservation")}})
    pd.DataFrame(rc_rows).to_csv(OUT / "risk_coverage.csv", index=False)

    agg = metrics(test)
    oracle_wer = 0.2132
    agg["gain_capture"] = (agg["a0_wer"] - agg["wer"]) / (agg["a0_wer"] - oracle_wer)
    per_speaker = {s: metrics(g) for s, g in test.groupby("speaker_id")}

    # D1-vs-D2 suppression diagnostics (D1 = tau 0)
    d1 = pd.read_csv(OUT / "d1_predictions.csv", keep_default_na=False, na_values=[""])
    d1_map = d1.set_index("sample_id")
    diag = {}
    for s, g in test.groupby("speaker_id"):
        d1_g = d1_map.loc[[i for i in g["sample_id"]]]
        d1_switch = d1_g[~d1_g["kept_a0"].astype(bool)]
        suppressed = g.set_index("sample_id").loc[d1_switch.index]
        suppressed = suppressed[~suppressed["switched"]]
        sup_out = d1_switch.loc[suppressed.index, "outcome"]
        diag[s] = {"d1_switches": int(len(d1_switch)),
                   "d2_suppressed": int(len(suppressed)),
                   "suppressed_were_improved": int((sup_out == "IMPROVED").sum()),
                   "suppressed_were_worsened": int((sup_out == "WORSENED").sum()),
                   "suppressed_were_unchanged": int((sup_out == "UNCHANGED").sum())}

    ctl_metrics = {"asr_wer": corpus_wer(ctl["ref"], ctl["a0"]),
                   "d2_wer_mean": float(np.mean(
                       [corpus_wer(g["ref"], g["d2_text"]) for _, g in ctl.groupby("fold")])),
                   "preservation_mean": float(np.mean(
                       [float((~g[g["ref"] == g["a0"]]["switched"]).mean())
                        for _, g in ctl.groupby("fold")])),
                   "preservation_worst": float(min(
                       float((~g[g["ref"] == g["a0"]]["switched"]).mean())
                       for _, g in ctl.groupby("fold")))}

    report = {"frozen_grid": TAUS, "folds": fold_report, "aggregate": agg,
              "per_speaker": per_speaker,
              "unseen_prompts": metrics(test[test["unseen_prompt"] == True]),  # noqa: E712
              "low": metrics(test[test["repairability"] == "LOW"]),
              "control": ctl_metrics, "suppression_diagnostics": diag,
              "comparison": {"A0": 0.3175, "B": 0.3225, "C": 0.2723,
                             "D1": 0.2826, "D2": agg["wer"], "oracle": oracle_wer}}
    (OUT / "d2_evaluation.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"\n=== D2 AGGREGATE ===")
    print(f"A0 {agg['a0_wer']:.4f} -> D2 {agg['wer']:.4f} | capture {agg['gain_capture']*100:.1f}%")
    print(f"exact {agg['exact']*100:.1f}% | switch {agg['switch_rate']*100:.1f}% | "
          f"uncertain {agg['uncertain_rate']*100:.1f}%")
    print(f"improved {agg['improved']} / worsened {agg['worsened']} "
          f"(ratio {agg['ratio']:.2f}) | edit precision {agg['edit_precision']*100:.1f}%")
    print(f"preservation {agg['preservation']*100:.2f}%")
    print(f"LOW {agg['low_a0_wer']:.4f} -> {agg['low_wer']:.4f} harmful {agg['low_harmful_switches']}")
    print(f"unseen {agg['unseen_a0_wer']:.4f} -> {agg['unseen_wer']:.4f} "
          f"+{agg['unseen_improved']}/-{agg['unseen_worsened']}")
    print(f"control: {ctl_metrics['asr_wer']:.4f} -> {ctl_metrics['d2_wer_mean']:.4f} "
          f"preservation mean {ctl_metrics['preservation_mean']*100:.2f}% "
          f"worst {ctl_metrics['preservation_worst']*100:.2f}%")
    print("\n=== PER SPEAKER ===")
    for s in sorted(per_speaker, key=lambda x: per_speaker[x]["a0_wer"]):
        b = per_speaker[s]
        tau = next(f["selected_tau"] for f in fold_report
                   if f["test_speaker"] == s)
        print(f"  {s} tau={tau} n={b['n']:>3} A0 {b['a0_wer']:.4f} -> D2 {b['wer']:.4f} "
              f"+{b['improved']}/-{b['worsened']} switch {b['switch_rate']*100:.0f}% "
              f"uncertain {b['uncertain_rate']*100:.0f}%")
    print("\n=== SUPPRESSION DIAGNOSTICS (D1 switches D2 suppressed) ===")
    for s, d in diag.items():
        print(f"  {s}: D1 switches {d['d1_switches']}, suppressed {d['d2_suppressed']} "
              f"(improved {d['suppressed_were_improved']}, worsened {d['suppressed_were_worsened']}, "
              f"unchanged {d['suppressed_were_unchanged']})")


if __name__ == "__main__":
    main()
