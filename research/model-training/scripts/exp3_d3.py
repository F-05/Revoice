"""D3 -- final selector development round. 21 frozen features, same recipe.

    python scripts/exp3_d3.py

STATUS OF THE DATA: the 683 headMic rows are DEVELOPMENT data for D3 (they have
been inspected repeatedly, including the feature audit). D3 numbers are
development results; D2's frozen LOSO numbers remain the cleanest historical
research baseline.

FROZEN BEFORE TRAINING:
  - Features: the 16 D2 features (byte-identical values, same order) + exactly
    5 audited additions: changed_support_count, max_ally_overlap,
    novel_word_flag, deletion_only_flag, score_within_list. Total 21.
  - Model: same conditional logit (shared scorer, softmax over the list),
    same seed/epochs/optimizer/standardization-per-fold as D1/D2.
  - Decision layer: the byte-identical frozen D2 tau procedure (validation
    speaker only, grid {0..0.9 step .05}, preservation>=99 -> ratio>=4:1 ->
    min WER -> largest tau).
  - CAPACITY ESCALATION RULE (predetermined): train the optional nonlinear
    scorer (ONE configuration: MLP with a single 16-unit tanh hidden layer,
    Adam lr 0.01, wd 1e-3, 500 epochs, seed 20260830, same 21 features,
    same folds, same tau procedure) ONLY IF the conditional-logit D3
    aggregate WER (tau-calibrated) is > 0.27. No other models, no sweeps.

Output must always be one of H1-H5 (asserted per row).
Writes results/d3/.
"""

from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_d1 import build_rows, overlap_f1, word_edit  # noqa: E402
from exp3_d2 import (apply_tau, build_control_rows, corpus_wer, metrics,  # noqa: E402
                     probabilities, select_tau)

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "results" / "d3"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES21_NEW = ["changed_support_count", "max_ally_overlap", "novel_word_flag",
                  "deletion_only_flag", "score_within_list"]
SEED = 20260830
EPOCHS, LR, WD = 300, 0.05, 1e-3
NL_EPOCHS, NL_LR, NL_HIDDEN = 500, 0.01, 16
ESCALATE_IF_WER_ABOVE = 0.27
ORACLE = 0.2132
IDX_IS_A0, IDX_CT2_SCORE, IDX_CT2_MISSING = 0, 4, 8  # positions in the 16


def augment(utts):
    """Append the 5 new features; the first 16 stay byte-identical."""
    for u in utts:
        texts, a0 = u["texts"], u["a0"]
        a0_set = set(a0.split())
        wsets = [set(t.split()) for t in texts]
        global_support = Counter(w for ws in wsets for w in ws)
        scores = [(f[IDX_CT2_SCORE] if f[IDX_CT2_MISSING] == 0.0 else None)
                  for f in u["features"]]
        ct2 = [s for s in scores if s is not None]
        lo, hi = (min(ct2), max(ct2)) if ct2 else (0.0, 0.0)
        spread = hi - lo
        for i, f in enumerate(u["features"]):
            words = texts[i].split()
            is_a0 = f[IDX_IS_A0] == 1.0
            changed = [w for w in words if w not in a0_set]
            deleted = [w for w in a0_set if w not in set(words)]
            others = [wsets[j] for j in range(len(texts)) if j != i]
            if changed:
                support_count = float(np.median(
                    [sum(w in o for o in others) for w in changed]))
            else:
                support_count = float(len(others)) if is_a0 else 0.0
            ally = float(max([overlap_f1(wsets[i], o) for o in others] or [1.0]))
            novel = float(any(global_support[w] == 1 for w in changed))
            deletion_only = float((not is_a0) and (not changed) and bool(deleted))
            swl = ((scores[i] - lo) / spread
                   if scores[i] is not None and spread > 0 else 0.0)
            f.extend([support_count, ally, novel, deletion_only, float(swl)])
    return utts


def train_fold21(train_utts, val_utts, seed=SEED):
    torch.manual_seed(seed)
    all_feats = np.array([f for u in train_utts for f in u["features"]], dtype=np.float64)
    mu, sd = all_feats.mean(0), all_feats.std(0)
    sd[sd < 1e-9] = 1.0
    dim = all_feats.shape[1]

    def tensors(utts):
        return [(torch.tensor((np.array(u["features"]) - mu) / sd,
                              dtype=torch.float32), u["label"]) for u in utts]

    train_t, val_t = tensors(train_utts), tensors(val_utts)
    w = torch.zeros(dim, requires_grad=True)
    opt = torch.optim.Adam([w], lr=LR, weight_decay=WD)
    best_val, best_w = float("inf"), w.detach().clone()
    for epoch in range(EPOCHS):
        opt.zero_grad()
        loss = torch.stack([torch.nn.functional.cross_entropy(
            (x @ w).unsqueeze(0), torch.tensor([y])) for x, y in train_t]).mean()
        loss.backward()
        opt.step()
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                v = torch.stack([torch.nn.functional.cross_entropy(
                    (x @ w).unsqueeze(0), torch.tensor([y]))
                    for x, y in val_t]).mean().item()
            if v < best_val - 1e-6:
                best_val, best_w = v, w.detach().clone()
    return {"kind": "cl", "w": best_w, "mu": mu, "sd": sd}


class MLP(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.h = torch.nn.Linear(dim, NL_HIDDEN)
        self.o = torch.nn.Linear(NL_HIDDEN, 1)

    def forward(self, x):
        return self.o(torch.tanh(self.h(x))).squeeze(-1)


def train_fold_nl(train_utts, val_utts):
    torch.manual_seed(SEED)
    all_feats = np.array([f for u in train_utts for f in u["features"]], dtype=np.float64)
    mu, sd = all_feats.mean(0), all_feats.std(0)
    sd[sd < 1e-9] = 1.0

    def tensors(utts):
        return [(torch.tensor((np.array(u["features"]) - mu) / sd,
                              dtype=torch.float32), u["label"]) for u in utts]

    train_t, val_t = tensors(train_utts), tensors(val_utts)
    net = MLP(all_feats.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=NL_LR, weight_decay=WD)
    best_val, best_state = float("inf"), copy.deepcopy(net.state_dict())
    for epoch in range(NL_EPOCHS):
        opt.zero_grad()
        loss = torch.stack([torch.nn.functional.cross_entropy(
            net(x).unsqueeze(0), torch.tensor([y])) for x, y in train_t]).mean()
        loss.backward()
        opt.step()
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                v = torch.stack([torch.nn.functional.cross_entropy(
                    net(x).unsqueeze(0), torch.tensor([y]))
                    for x, y in val_t]).mean().item()
            if v < best_val - 1e-6:
                best_val, best_state = v, copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    return {"kind": "nl", "net": net, "mu": mu, "sd": sd}


def score_utts(model, utts):
    if model["kind"] == "cl":
        return probabilities(utts, model["w"], model["mu"], model["sd"])
    out = []
    for u in utts:
        x = torch.tensor((np.array(u["features"]) - model["mu"]) / model["sd"],
                         dtype=torch.float32)
        with torch.no_grad():
            p = torch.softmax(model["net"](x), dim=0).numpy()
        a0_idx = next((i for i, t in enumerate(u["texts"]) if t == u["a0"]), None)
        p_a0 = float(p[a0_idx]) if a0_idx is not None else 0.0
        alt = [(float(p[i]), i) for i in range(len(u["texts"])) if i != a0_idx]
        p_alt, alt_idx = max(alt) if alt else (0.0, None)
        out.append({"u": u, "p_a0": p_a0, "p_alt": p_alt, "alt_idx": alt_idx,
                    "margin": p_alt - p_a0, "argmax_is_a0": p_a0 >= p_alt})
    return out


def run_loso(utts, control, folds, trainer, tag):
    by_id = {u["sample_id"]: u for u in utts}
    tests, ctls, meta = [], [], []
    for fold in sorted(folds["fold"].unique()):
        fr = folds[folds["fold"] == fold]
        get = lambda role: [by_id[s] for s in fr[fr["role"] == role]["sample_id"]
                            if s in by_id]
        train_u, val_u, test_u = get("train"), get("validation"), get("test")
        model = trainer(train_u, val_u)
        tau, _ = select_tau(score_utts(model, val_u))
        tdf = apply_tau(score_utts(model, test_u), tau)
        tdf["fold"] = fold
        # hard constraint check
        for _, r in tdf.iterrows():
            assert r["d2_text"] in by_id[r["sample_id"]]["texts"]
        cdf = apply_tau(score_utts(model, control), tau)
        cdf["fold"] = fold
        tests.append(tdf); ctls.append(cdf)
        entry = {"fold": int(fold), "test_speaker": test_u[0]["speaker_id"], "tau": tau}
        if model["kind"] == "cl":
            entry["weights"] = [round(float(v), 4) for v in model["w"]]
            entry["mu"] = [round(float(v), 4) for v in model["mu"]]
            entry["sd"] = [round(float(v), 4) for v in model["sd"]]
        meta.append(entry)
        print(f"  [{tag}] fold {fold} ({test_u[0]['speaker_id']}) tau={tau}", flush=True)
    return pd.concat(tests, ignore_index=True), pd.concat(ctls, ignore_index=True), meta


def summarize(test, ctl, tag):
    agg = metrics(test)
    agg["gain_capture"] = (agg["a0_wer"] - agg["wer"]) / (agg["a0_wer"] - ORACLE)
    per_speaker = {s: metrics(g) for s, g in test.groupby("speaker_id")}
    ctl_stats = {
        "asr_wer": corpus_wer(ctl["ref"], ctl["a0"]),
        "wer_mean": float(np.mean([corpus_wer(g["ref"], g["d2_text"])
                                   for _, g in ctl.groupby("fold")])),
        "preservation_mean": float(np.mean(
            [float((~g[g["ref"] == g["a0"]]["switched"]).mean())
             for _, g in ctl.groupby("fold")])),
        "preservation_worst": float(min(
            float((~g[g["ref"] == g["a0"]]["switched"]).mean())
            for _, g in ctl.groupby("fold")))}
    print(f"\n=== {tag} aggregate ===")
    print(f"A0 {agg['a0_wer']:.4f} -> {agg['wer']:.4f} | capture {agg['gain_capture']*100:.1f}% "
          f"| exact {agg['exact']*100:.1f}%")
    print(f"+{agg['improved']}/-{agg['worsened']} (ratio {agg['ratio']:.2f}) | "
          f"precision {agg['edit_precision']*100:.1f}% | switch {agg['switch_rate']*100:.1f}%")
    print(f"preservation {agg['preservation']*100:.2f}% | control pres "
          f"{ctl_stats['preservation_mean']*100:.2f}% (worst {ctl_stats['preservation_worst']*100:.2f}%)")
    print(f"LOW {agg['low_a0_wer']:.4f}->{agg['low_wer']:.4f} harmful {agg['low_harmful_switches']} "
          f"| unseen {agg['unseen_a0_wer']:.4f}->{agg['unseen_wer']:.4f} "
          f"+{agg['unseen_improved']}/-{agg['unseen_worsened']}")
    return {"aggregate": agg, "per_speaker": per_speaker, "control": ctl_stats}


def main() -> None:
    utts = augment(build_rows())
    control = augment(build_control_rows())
    folds = pd.read_csv(PROJECT / "data/large_torgo/loso_folds.csv")
    print(f"{len(utts)} dev utterances, {len(utts[0]['features'][0])} features")

    print("\n--- D3 conditional logit (21 features, 21 parameters) ---")
    test_cl, ctl_cl, meta_cl = run_loso(utts, control, folds, train_fold21, "CL")
    cl = summarize(test_cl, ctl_cl, "D3-CL")
    test_cl.to_csv(OUT / "d3_predictions.csv", index=False)
    ctl_cl.to_csv(OUT / "d3_control_predictions.csv", index=False)

    result = {"features_new": FEATURES21_NEW, "cl": cl, "fold_meta": meta_cl,
              "escalation_rule": f"NL only if CL WER > {ESCALATE_IF_WER_ABOVE}",
              "comparison": {"A0": 0.3175, "C": 0.2723, "D2": 0.2849,
                             "oracle": ORACLE, "D3_CL": cl["aggregate"]["wer"]}}

    if cl["aggregate"]["wer"] > ESCALATE_IF_WER_ABOVE:
        print("\n--- escalation triggered: D3-NL (predetermined single config) ---")
        test_nl, ctl_nl, meta_nl = run_loso(utts, control, folds,
                                            train_fold_nl, "NL")
        nl = summarize(test_nl, ctl_nl, "D3-NL")
        test_nl.to_csv(OUT / "d3nl_predictions.csv", index=False)
        ctl_nl.to_csv(OUT / "d3nl_control_predictions.csv", index=False)
        result["nl"] = nl
        result["comparison"]["D3_NL"] = nl["aggregate"]["wer"]
    else:
        print("\nEscalation NOT triggered (CL <= 0.27); conditional logit stands.")

    (OUT / "d3_evaluation.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote {OUT}/")


if __name__ == "__main__":
    main()
