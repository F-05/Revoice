"""Deployment Step 1+2: global OOF tau + final D3-NL product selector.

    python scripts/deploy_calibrate.py

Step 1: deterministically re-derive the 8 frozen D3-NL fold models (identical
seed/recipe), pool every utterance's OUT-OF-FOLD scored prediction, sweep the
frozen tau grid once, and select ONE global tau by the frozen priority:
preservation>=99 -> ratio>=4 -> precision>=75 -> LOW safe -> min WER ->
largest tau. Control preservation checked with per-fold models at the same tau.

Step 2: train the final product selector on all 683 rows with the identical
frozen NL recipe, running for the MEDIAN of the 8 folds' best-checkpoint epochs
(deterministic; no new tuning; no held-out set consumed).

Writes models/revoice_selector_v1.json and results/d3/deployment_calibration.json.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp3_d1 import FEATURES as FEATURES16  # noqa: E402
from exp3_d2 import TAUS, apply_tau, build_control_rows, corpus_wer, metrics  # noqa: E402
from exp3_d3 import (MLP, NL_EPOCHS, NL_LR, SEED, WD, augment,  # noqa: E402
                     build_rows, score_utts)

PROJECT = Path(__file__).resolve().parent.parent
OUT_MODEL = PROJECT / "models" / "revoice_selector_v1.json"
OUT_CAL = PROJECT / "results" / "d3" / "deployment_calibration.json"
FEATURES21 = FEATURES16 + ["changed_support_count", "max_ally_overlap",
                           "novel_word_flag", "deletion_only_flag",
                           "score_within_list"]


def train_nl(train_utts, val_utts):
    """Frozen NL recipe; also returns the best-checkpoint epoch."""
    torch.manual_seed(SEED)
    feats = np.array([f for u in train_utts for f in u["features"]], dtype=np.float64)
    mu, sd = feats.mean(0), feats.std(0)
    sd[sd < 1e-9] = 1.0

    def tensors(utts):
        return [(torch.tensor((np.array(u["features"]) - mu) / sd,
                              dtype=torch.float32), u["label"]) for u in utts]

    train_t = tensors(train_utts)
    val_t = tensors(val_utts) if val_utts else None
    net = MLP(feats.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=NL_LR, weight_decay=WD)
    best_val, best_state, best_epoch = float("inf"), copy.deepcopy(net.state_dict()), NL_EPOCHS
    for epoch in range(NL_EPOCHS):
        opt.zero_grad()
        loss = torch.stack([torch.nn.functional.cross_entropy(
            net(x).unsqueeze(0), torch.tensor([y])) for x, y in train_t]).mean()
        loss.backward()
        opt.step()
        if val_t and (epoch + 1) % 10 == 0:
            with torch.no_grad():
                v = torch.stack([torch.nn.functional.cross_entropy(
                    net(x).unsqueeze(0), torch.tensor([y]))
                    for x, y in val_t]).mean().item()
            if v < best_val - 1e-6:
                best_val, best_state, best_epoch = v, copy.deepcopy(net.state_dict()), epoch + 1
    if val_t:
        net.load_state_dict(best_state)
    return net, mu, sd, best_epoch


def train_nl_epochs(train_utts, epochs):
    torch.manual_seed(SEED)
    feats = np.array([f for u in train_utts for f in u["features"]], dtype=np.float64)
    mu, sd = feats.mean(0), feats.std(0)
    sd[sd < 1e-9] = 1.0
    tensors = [(torch.tensor((np.array(u["features"]) - mu) / sd,
                             dtype=torch.float32), u["label"]) for u in train_utts]
    net = MLP(feats.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=NL_LR, weight_decay=WD)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.stack([torch.nn.functional.cross_entropy(
            net(x).unsqueeze(0), torch.tensor([y])) for x, y in tensors]).mean()
        loss.backward()
        opt.step()
    return net, mu, sd


def main() -> None:
    utts = augment(build_rows())
    control = augment(build_control_rows())
    folds = pd.read_csv(PROJECT / "data/large_torgo/loso_folds.csv")
    by_id = {u["sample_id"]: u for u in utts}

    # --- Step 1: OOF pooled predictions -----------------------------------
    pooled, ctl_by_fold, best_epochs = [], {}, []
    frozen = pd.read_csv(PROJECT / "results/d3/d3nl_predictions.csv",
                         keep_default_na=False, na_values=[""]).set_index("sample_id")
    for fold in sorted(folds["fold"].unique()):
        fr = folds[folds["fold"] == fold]
        get = lambda role: [by_id[s] for s in fr[fr["role"] == role]["sample_id"]
                            if s in by_id]
        net, mu, sd, be = train_nl(get("train"), get("validation"))
        best_epochs.append(be)
        model = {"kind": "nl", "net": net, "mu": mu, "sd": sd}
        scored = score_utts(model, get("test"))
        # determinism check against the frozen run's stored margins
        for s in scored[:3]:
            stored = frozen.loc[s["u"]["sample_id"], "margin"]
            assert abs(stored - s["margin"]) < 1e-5, "retrain does not match frozen run"
        pooled.extend(scored)
        ctl_by_fold[fold] = score_utts(model, control)
        print(f"fold {fold}: OK (best epoch {be})", flush=True)

    print(f"\npooled OOF predictions: {len(pooled)}")
    stats = []
    for tau in TAUS:
        df = apply_tau(pooled, tau)
        m = metrics(df)
        m["tau"] = tau
        m["n_switches"] = int(df["switched"].sum())
        ctl_pres = []
        for fold, cs in ctl_by_fold.items():
            cdf = apply_tau(cs, tau)
            correct = cdf[cdf["ref"] == cdf["a0"]]
            ctl_pres.append(float((~correct["switched"]).mean()))
        m["control_preservation_mean"] = float(np.mean(ctl_pres))
        stats.append(m)

    ok = [m for m in stats if m["preservation"] is not None and m["preservation"] >= 0.99]
    ok = [m for m in ok if m["worsened"] == 0 or m["ratio"] >= 4.0] or ok
    ok = [m for m in ok if m["edit_precision"] is None or m["edit_precision"] >= 0.75] or ok
    ok = [m for m in ok if m["low_wer"] is None or m["low_wer"] <= m["low_a0_wer"] + 1e-9] or ok
    best_wer = min(m["wer"] for m in ok)
    chosen = max([m for m in ok if m["wer"] <= best_wer + 1e-9], key=lambda m: m["tau"])
    tau_global = chosen["tau"]

    print("\n=== FULL GRID (pooled OOF) ===")
    print(f"{'tau':>5} {'WER':>7} {'switch':>7} {'impr':>5} {'wors':>5} {'prec':>6} {'pres':>7} {'ctl_pres':>8}")
    for m in stats:
        prec = "n/a" if m["edit_precision"] is None else f"{m['edit_precision']*100:.0f}%"
        print(f"{m['tau']:>5} {m['wer']:>7.4f} {m['n_switches']:>7} {m['improved']:>5} "
              f"{m['worsened']:>5} {prec:>6} {m['preservation']*100:>6.2f}% "
              f"{m['control_preservation_mean']*100:>7.2f}%")

    final_df = apply_tau(pooled, tau_global)
    final_m = metrics(final_df)
    final_m["gain_capture"] = (final_m["a0_wer"] - final_m["wer"]) / (final_m["a0_wer"] - 0.2132)
    per_speaker = {s: metrics(g) for s, g in final_df.groupby("speaker_id")}
    print(f"\nGLOBAL TAU = {tau_global}")
    print(f"OOF: A0 {final_m['a0_wer']:.4f} -> {final_m['wer']:.4f} | "
          f"+{final_m['improved']}/-{final_m['worsened']} (ratio {final_m['ratio']:.2f}) | "
          f"precision {'n/a' if final_m['edit_precision'] is None else round(final_m['edit_precision']*100,1)} | pres {final_m['preservation']*100:.2f}% | "
          f"ctl pres {chosen['control_preservation_mean']*100:.2f}%")
    print(f"LOW {final_m['low_a0_wer']:.4f}->{final_m['low_wer']:.4f} | "
          f"unseen {final_m['unseen_a0_wer']:.4f}->{final_m['unseen_wer']:.4f} | "
          f"switch {final_m['switch_rate']*100:.1f}%")
    for s in sorted(per_speaker, key=lambda x: per_speaker[x]["a0_wer"]):
        b = per_speaker[s]
        print(f"  {s} n={b['n']:>3} {b['a0_wer']:.4f} -> {b['wer']:.4f} "
              f"+{b['improved']}/-{b['worsened']}")

    (OUT_CAL).write_text(json.dumps(
        {"tau_global": tau_global, "grid": stats, "oof_metrics": final_m,
         "per_speaker": per_speaker, "fold_best_epochs": best_epochs},
        indent=2, default=str))

    # --- Step 2: final product model --------------------------------------
    final_epochs = int(np.median(best_epochs))
    print(f"\nfinal model: training on all {len(utts)} rows for {final_epochs} epochs "
          f"(median of fold best epochs {best_epochs})")
    net, mu, sd = train_nl_epochs(utts, final_epochs)

    artifact = {
        "model_type": "conditional-mlp-selector",
        "model_version": "revoice_selector_v1",
        "feature_version": "d3-21f-v1",
        "features": FEATURES21,
        "scaler_mean": [float(v) for v in mu],
        "scaler_std": [float(v) for v in sd],
        "W1": net.h.weight.detach().numpy().tolist(),
        "b1": net.h.bias.detach().numpy().tolist(),
        "W2": net.o.weight.detach().numpy().squeeze(0).tolist(),
        "b2": float(net.o.bias.detach().numpy()[0]),
        "hidden_activation": "tanh",
        "tau_global": tau_global,
        "training_epochs": final_epochs,
        "seed": SEED,
        "whisper_model": "medium.en",
        "nbest": {"beam_size": 12, "num_hypotheses": 8, "keep_unique": 5,
                  "vad": "silero-default", "loop_collapse": ">=2-word phrase repeated >=2x",
                  "dedupe": "normalized text"},
        "hybrid_list": "H1=production A0; H2..H5=ct2 top-4 unique (deduped vs A0)",
        "normalization": "NFKC, lowercase, strip punct (keep intra-word ' and -), collapse ws",
        "provenance": {"research_baseline": "D2 frozen LOSO (WER 0.2849)",
                       "product_selector": "D3-NL development-stage (OOF-calibrated)",
                       "trained_on": "683 TORGO headMic dysarthric sentences",
                       "date": "2026-08-30"},
    }
    OUT_MODEL.write_text(json.dumps(artifact, indent=2))
    print(f"wrote {OUT_MODEL} ({OUT_MODEL.stat().st_size} bytes)")

    # sanity: numpy-only inference reproduces torch on a sample
    x = (np.array(utts[0]["features"]) - mu) / sd
    np_scores = np.tanh(x @ np.array(artifact["W1"]).T + artifact["b1"]) @ np.array(artifact["W2"]) + artifact["b2"]
    with torch.no_grad():
        t_scores = net(torch.tensor(x, dtype=torch.float32)).numpy()
    assert np.allclose(np_scores, t_scores, atol=1e-5), "numpy inference mismatch"
    print("numpy-only inference verified against torch")


if __name__ == "__main__":
    main()
