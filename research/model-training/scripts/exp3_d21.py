"""Experiment 3, D2.1 -- speaker-robust conservative calibration (nested).

    python scripts/exp3_d21.py

Model, features, hybrid lists, outer LOSO folds: IDENTICAL to D1/D2 (frozen).
Only the calibration policy changes.

NESTED PROCEDURE (per outer fold k with test speaker T_k):
  inner speakers = the 7 speakers != T_k
  for each inner speaker s:
     early-stop speaker = difficulty-rank neighbour of s within the inner set
     train the 16-param selector on the remaining 5 inner speakers
     score s's utterances (a model that never saw s OR T_k)
  pool the 7 inner speakers' rows; evaluate the FROZEN grid
  tau in {0.00,0.05,...,0.90} and select the FLOOR by frozen priority:
     A. pooled correct-input preservation >= 99%
        (if none: keep taus with maximal preservation)
     B. improved:worsened >= 4:1 where meaningful (>=5 pooled switches);
        skip if no tau qualifies
     C. LOW safety: pooled LOW WER must not exceed pooled LOW A0 WER
     D. minimize pooled WER
     E. tie-break: LARGEST tau
  tau_final(k) = max(tau_val(k), floor(k))
     where tau_val(k) is D2's validation-speaker threshold, recomputed with
     the identical frozen procedure.
  Evaluate T_k exactly once at tau_final(k).

LEAKAGE PROOF: floor(k) is a deterministic function of (a) the 7 non-test
speakers' audio-derived features and references, (b) the frozen grid, (c) the
frozen priority. T_k's rows appear in no inner training set, no inner
early-stopping set, and no inner evaluation pool; no outer-test WER,
preservation, unseen-prompt, LOW, or switch outcome is read before tau_final
is fixed. tau_val(k) uses only the outer validation speaker (never T_k).

Writes results/experiment3_selector/{d21_predictions.csv,
d21_control_predictions.csv,d21_evaluation.json}. D1/D2 artifacts untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_d1 import build_rows, train_fold  # noqa: E402
from exp3_d2 import (TAUS, apply_tau, build_control_rows, corpus_wer,  # noqa: E402
                     metrics, probabilities, select_tau)

PROJECT = Path(__file__).resolve().parent.parent
LT = PROJECT / "data" / "large_torgo"
OUT = PROJECT / "results" / "experiment3_selector"

# measured A0 difficulty order (ascending WER), frozen since experiment 2
DIFFICULTY = ["M03", "F04", "F03", "M05", "M02", "F01", "M01", "M04"]


def inner_neighbour(s: str, pool: list[str]) -> str:
    """Difficulty-rank neighbour of s among pool: next harder, else next easier."""
    harder = [x for x in DIFFICULTY[DIFFICULTY.index(s) + 1:] if x in pool]
    if harder:
        return harder[0]
    easier = [x for x in DIFFICULTY[:DIFFICULTY.index(s)] if x in pool]
    return easier[-1]


def select_floor(pooled_scored):
    stats = []
    for tau in TAUS:
        df = apply_tau(pooled_scored, tau)
        m = metrics(df)
        m["tau"] = tau
        m["n_switches"] = int(df["switched"].sum())
        stats.append(m)
    ok = [m for m in stats if (m["preservation"] is None or m["preservation"] >= 0.99)]
    if not ok:
        best = max(m["preservation"] for m in stats)
        ok = [m for m in stats if m["preservation"] == best]
    ratio_ok = [m for m in ok if m["n_switches"] >= 5 and m["ratio"] >= 4.0]
    pool = ratio_ok if ratio_ok else ok
    low_ok = [m for m in pool
              if m["low_wer"] is None or m["low_a0_wer"] is None
              or m["low_wer"] <= m["low_a0_wer"] + 1e-9]
    pool = low_ok if low_ok else pool
    best_wer = min(m["wer"] for m in pool)
    tied = [m for m in pool if m["wer"] <= best_wer + 1e-9]
    return max(tied, key=lambda m: m["tau"])["tau"], stats


def main() -> None:
    utts = build_rows()
    control = build_control_rows()
    folds = pd.read_csv(LT / "loso_folds.csv")
    by_id = {u["sample_id"]: u for u in utts}
    by_speaker = {}
    for u in utts:
        by_speaker.setdefault(u["speaker_id"], []).append(u)
    speakers = sorted(by_speaker, key=DIFFICULTY.index)
    print(f"{len(utts)} utterances, speakers {speakers}")

    all_test, all_ctl, fold_meta = [], [], []
    for fold in sorted(folds["fold"].unique()):
        fr = folds[folds["fold"] == fold]
        get = lambda role: [by_id[s] for s in fr[fr["role"] == role]["sample_id"]
                            if s in by_id]
        train_u, val_u, test_u = get("train"), get("validation"), get("test")
        t_speaker = test_u[0]["speaker_id"]
        inner_pool = [s for s in speakers if s != t_speaker]

        # --- inner LOSO over the 7 non-test speakers (T_k fully excluded) ---
        pooled = []
        for s in inner_pool:
            stop = inner_neighbour(s, [x for x in inner_pool if x != s])
            inner_train = [u for sp in inner_pool if sp not in (s, stop)
                           for u in by_speaker[sp]]
            w_i, mu_i, sd_i = train_fold(inner_train, by_speaker[stop])
            pooled.extend(probabilities(by_speaker[s], w_i, mu_i, sd_i))
        floor, _ = select_floor(pooled)

        # --- outer model + D2's validation tau (identical frozen procedure) --
        w, mu, sd = train_fold(train_u, val_u)
        tau_val, _ = select_tau(probabilities(val_u, w, mu, sd))
        tau_final = max(tau_val, floor)

        test_df = apply_tau(probabilities(test_u, w, mu, sd), tau_final)
        test_df["fold"] = fold
        all_test.append(test_df)
        ctl_df = apply_tau(probabilities(control, w, mu, sd), tau_final)
        ctl_df["fold"] = fold
        all_ctl.append(ctl_df)

        fold_meta.append({"fold": int(fold), "test_speaker": t_speaker,
                          "tau_val": tau_val, "floor": floor,
                          "tau_final": tau_final})
        print(f"fold {fold} ({t_speaker}): tau_val={tau_val} floor={floor} "
              f"-> tau_final={tau_final}")

    test = pd.concat(all_test, ignore_index=True)
    ctl = pd.concat(all_ctl, ignore_index=True)
    test.to_csv(OUT / "d21_predictions.csv", index=False)
    ctl.to_csv(OUT / "d21_control_predictions.csv", index=False)

    agg = metrics(test)
    oracle_wer = 0.2132
    agg["gain_capture"] = (agg["a0_wer"] - agg["wer"]) / (agg["a0_wer"] - oracle_wer)
    per_speaker = {s: metrics(g) for s, g in test.groupby("speaker_id")}

    ctl_stats = {"asr_wer": corpus_wer(ctl["ref"], ctl["a0"]),
                 "d21_wer_mean": float(np.mean([corpus_wer(g["ref"], g["d2_text"])
                                                for _, g in ctl.groupby("fold")])),
                 "preservation_mean": float(np.mean(
                     [float((~g[g["ref"] == g["a0"]]["switched"]).mean())
                      for _, g in ctl.groupby("fold")])),
                 "preservation_worst": float(min(
                     float((~g[g["ref"] == g["a0"]]["switched"]).mean())
                     for _, g in ctl.groupby("fold")))}

    # --- D2 vs D2.1 suppression diagnostics -------------------------------
    d2 = pd.read_csv(OUT / "d2_predictions.csv", keep_default_na=False,
                     na_values=[""]).set_index("sample_id")
    t_idx = test.set_index("sample_id")
    d2_switch = d2[d2["switched"].astype(bool)]
    suppressed = t_idx.loc[d2_switch.index]
    suppressed = suppressed[~suppressed["switched"]]
    sup_out = d2_switch.loc[suppressed.index, "outcome"]
    suppression = {"d2_switches": int(len(d2_switch)),
                   "d21_suppressed": int(len(suppressed)),
                   "suppressed_were_improved": int((sup_out == "IMPROVED").sum()),
                   "suppressed_were_worsened": int((sup_out == "WORSENED").sum()),
                   "suppressed_were_unchanged": int((sup_out == "UNCHANGED").sum()),
                   "per_speaker": {s: int((~t_idx.loc[g.index, "switched"]).sum())
                                   for s, g in d2_switch.groupby(
                                       d2_switch["speaker_id"])}}

    report = {"procedure": "nested speaker-robust floor; tau_final=max(tau_val,floor)",
              "folds": fold_meta, "aggregate": agg, "per_speaker": per_speaker,
              "unseen_prompts": metrics(test[test["unseen_prompt"] == True]),  # noqa: E712
              "low": metrics(test[test["repairability"] == "LOW"]),
              "control": ctl_stats, "d2_vs_d21_suppression": suppression,
              "comparison": {"A0": 0.3175, "B": 0.3225, "C": 0.2723,
                             "D1": 0.2826, "D2": 0.2849, "D2.1": agg["wer"],
                             "oracle": oracle_wer}}
    (OUT / "d21_evaluation.json").write_text(json.dumps(report, indent=2, default=str))

    u, lo = report["unseen_prompts"], report["low"]
    print(f"\n=== D2.1 AGGREGATE ===")
    print(f"A0 {agg['a0_wer']:.4f} -> D2.1 {agg['wer']:.4f} | capture {agg['gain_capture']*100:.1f}%")
    print(f"exact {agg['exact']*100:.1f}% | switch {agg['switch_rate']*100:.1f}% | "
          f"keep {agg['keep_a0_rate']*100:.1f}% | uncertain {agg['uncertain_rate']*100:.1f}%")
    print(f"improved {agg['improved']} / worsened {agg['worsened']} "
          f"(ratio {agg['ratio']:.2f}) | edit precision {agg['edit_precision']*100:.1f}%")
    print(f"preservation {agg['preservation']*100:.2f}% | control preservation "
          f"{ctl_stats['preservation_mean']*100:.2f}% (worst {ctl_stats['preservation_worst']*100:.2f}%)")
    print(f"LOW {lo['a0_wer']:.4f} -> {lo['wer']:.4f} +{lo['improved']}/-{lo['worsened']}")
    print(f"unseen {u['a0_wer']:.4f} -> {u['wer']:.4f} +{u['improved']}/-{u['worsened']}")
    print("\n=== PER SPEAKER ===")
    for s in sorted(per_speaker, key=lambda x: per_speaker[x]["a0_wer"]):
        b = per_speaker[s]
        fm = next(f for f in fold_meta if f["test_speaker"] == s)
        print(f"  {s} tau_val={fm['tau_val']} floor={fm['floor']} final={fm['tau_final']} "
              f"n={b['n']:>3} A0 {b['a0_wer']:.4f} -> {b['wer']:.4f} "
              f"+{b['improved']}/-{b['worsened']} switch {b['switch_rate']*100:.0f}%")
    print("\n=== D2 -> D2.1 SUPPRESSION ===")
    for k, v in suppression.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
