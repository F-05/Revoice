"""Experiment 3, Stage 3 -- train and evaluate D1 (pure hypothesis selector).

    python scripts/exp3_d1.py

Conditional-logit selector: score(Hi) = w . x_i, shared w, softmax over the
utterance's hybrid list. Trained per LOSO fold on train-speaker rows, early
stopping on the fold's validation speaker, evaluated once on the test speaker.
Output ALWAYS equals one of H1-H5 (hard constraint, verified).

No thresholds (that is D2). No control set yet. Reference used only to build
training labels; never at inference.

Writes results/experiment3_selector/{d1_predictions.csv,d1_evaluation.json}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT = Path(__file__).resolve().parent.parent
LT = PROJECT / "data" / "large_torgo"
OUT = PROJECT / "results" / "experiment3_selector"

TIERS = {"M03": "easy", "F04": "easy", "F03": "medium", "M05": "medium",
         "M02": "hard", "F01": "hard", "M01": "hard", "M04": "hard"}
SEED = 20260830
EPOCHS = 300
LR = 0.05
WEIGHT_DECAY = 1e-3

FEATURES = [
    "is_a0", "rank",
    # A0-specific family (nonzero only on the A0 candidate)
    "a0_conf", "a0_conf_missing",
    # ct2-specific family (nonzero only on ct2 candidates)
    "ct2_score", "ct2_score_rel_best", "ct2_score_gap_next",
    "ct2_score_per_word", "ct2_score_missing",
    # comparable cross-hypothesis evidence
    "edit_to_a0", "edit_to_a0_norm", "n_words", "len_ratio_median",
    "consensus_f1", "support_frac_ge2", "disputed_support",
]


def word_edit(a: list[str], b: list[str]) -> int:
    """Word-level Levenshtein distance."""
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[n]


def overlap_f1(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return 2 * inter / (len(a) + len(b))


def build_rows():
    pairs = pd.read_csv(LT / "repair_pairs.csv", keep_default_na=False, na_values=[""])
    for c in ("asr_transcript_normalized", "ground_truth_normalized"):
        pairs[c] = pairs[c].fillna("")
    oracle = pd.read_csv(OUT / "oracle_rows.csv", keep_default_na=False, na_values=[""])
    meta = pairs.set_index("sample_id")

    nbest = {}
    for line in (LT / "nbest" / "nbest_cache.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            nbest[r["sample_id"]] = r["hypotheses"]  # [{processed, score, raw}]

    utterances = []
    for _, orow in oracle.iterrows():
        sid = orow["sample_id"]
        m = meta.loc[sid]
        a0 = m["asr_transcript_normalized"]
        a0_conf = m["asr_confidence"] if pd.notna(m["asr_confidence"]) else None

        cands, seen = [], set()
        if a0:
            cands.append({"text": a0, "is_a0": 1, "ct2_score": None})
            seen.add(a0)
        for h in nbest[sid]:
            t = h["processed"]
            if t and t not in seen:
                seen.add(t)
                cands.append({"text": t, "is_a0": 0, "ct2_score": h["score"]})
            if len(cands) >= 5:
                break
        if not cands:
            cands = [{"text": "", "is_a0": 1, "ct2_score": None}]

        ct2_scores = [c["ct2_score"] for c in cands if c["ct2_score"] is not None]
        best_ct2 = max(ct2_scores) if ct2_scores else 0.0
        sorted_ct2 = sorted(ct2_scores, reverse=True)
        lengths = [len(c["text"].split()) for c in cands]
        median_len = float(np.median([l for l in lengths if l > 0]) or 1.0)
        word_sets = [set(c["text"].split()) for c in cands]
        a0_words = a0.split()
        a0_set = set(a0_words)
        # words attested across candidates
        from collections import Counter
        support = Counter(w for ws in word_sets for w in ws)

        feats = []
        for i, c in enumerate(cands):
            words = c["text"].split()
            wset = word_sets[i]
            others = [word_sets[j] for j in range(len(cands)) if j != i]
            cons = float(np.mean([overlap_f1(wset, o) for o in others])) if others else 1.0
            sup2 = (np.mean([support[w] >= 3 for w in words]) if words else 0.0)
            # support counts include self, so >=3 means the word appears in
            # >=2 OTHER hypotheses
            disputed = [w for w in words if w not in a0_set]
            dsup = (float(np.mean([support[w] >= 2 for w in disputed]))
                    if disputed else 1.0)
            ed = word_edit(words, a0_words)
            score = c["ct2_score"]
            if score is not None and len(sorted_ct2) > 1:
                pos = sorted_ct2.index(score)
                gap = (sorted_ct2[pos] - sorted_ct2[pos + 1]
                       if pos + 1 < len(sorted_ct2) else 0.0)
            else:
                gap = 0.0
            feats.append({
                "is_a0": float(c["is_a0"]),
                "rank": float(i),
                "a0_conf": float(a0_conf) if (c["is_a0"] and a0_conf is not None) else 0.0,
                "a0_conf_missing": float(c["is_a0"] and a0_conf is None),
                "ct2_score": float(score) if score is not None else 0.0,
                "ct2_score_rel_best": float(score - best_ct2) if score is not None else 0.0,
                "ct2_score_gap_next": float(gap),
                "ct2_score_per_word": (float(score) / max(len(words), 1)
                                       if score is not None else 0.0),
                "ct2_score_missing": float(score is None),
                "edit_to_a0": float(ed),
                "edit_to_a0_norm": float(ed) / max(len(words), len(a0_words), 1),
                "n_words": float(len(words)),
                "len_ratio_median": float(len(words)) / median_len if median_len else 0.0,
                "consensus_f1": cons,
                "support_frac_ge2": float(sup2),
                "disputed_support": dsup,
            })
        utterances.append({
            "sample_id": sid, "speaker_id": orow["speaker_id"],
            "tier": orow["tier"], "repairability": orow["repairability"],
            "unseen_prompt": bool(orow["unseen_prompt"]),
            "ref": orow["ref"], "a0": a0,
            "texts": [c["text"] for c in cands],
            "features": [[f[k] for k in FEATURES] for f in feats],
            "label": int(orow["oracle_idx"]),
            "oracle_text": orow["oracle_text"],
            "a0_wer": float(orow["a0_wer"]), "oracle_wer": float(orow["oracle_wer"]),
        })
    return utterances


def train_fold(train_utts, val_utts):
    torch.manual_seed(SEED)
    dim = len(FEATURES)
    # standardize over all train candidate rows
    all_feats = np.array([f for u in train_utts for f in u["features"]], dtype=np.float64)
    mu, sd = all_feats.mean(0), all_feats.std(0)
    sd[sd < 1e-9] = 1.0

    def tensors(utts):
        return [(torch.tensor((np.array(u["features"]) - mu) / sd, dtype=torch.float32),
                 u["label"]) for u in utts]

    train_t, val_t = tensors(train_utts), tensors(val_utts)
    w = torch.zeros(dim, requires_grad=True)
    opt = torch.optim.Adam([w], lr=LR, weight_decay=WEIGHT_DECAY)
    best_val, best_w = float("inf"), w.detach().clone()
    for epoch in range(EPOCHS):
        opt.zero_grad()
        loss = torch.stack([torch.nn.functional.cross_entropy(
            (x @ w).unsqueeze(0), torch.tensor([y])) for x, y in train_t]).mean()
        loss.backward()
        opt.step()
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                vloss = torch.stack([torch.nn.functional.cross_entropy(
                    (x @ w).unsqueeze(0), torch.tensor([y]))
                    for x, y in val_t]).mean().item()
            if vloss < best_val - 1e-6:
                best_val, best_w = vloss, w.detach().clone()
    return best_w, mu, sd


def main() -> None:
    utts = build_rows()
    folds = pd.read_csv(LT / "loso_folds.csv")
    by_id = {u["sample_id"]: u for u in utts}
    print(f"{len(utts)} utterances, {len(FEATURES)} features -> "
          f"trainable parameters: {len(FEATURES)}")

    rows, fold_stats = [], []
    for fold in sorted(folds["fold"].unique()):
        fr = folds[folds["fold"] == fold]
        get = lambda role: [by_id[s] for s in fr[fr["role"] == role]["sample_id"]
                            if s in by_id]
        train_u, val_u, test_u = get("train"), get("validation"), get("test")
        w, mu, sd = train_fold(train_u, val_u)

        for u in test_u:
            x = torch.tensor((np.array(u["features"]) - mu) / sd, dtype=torch.float32)
            with torch.no_grad():
                probs = torch.softmax(x @ w, dim=0).numpy()
            pick = int(probs.argmax())
            picked_text = u["texts"][pick]
            assert picked_text in u["texts"]  # hard constraint
            rows.append({
                "fold": fold, "sample_id": u["sample_id"],
                "speaker_id": u["speaker_id"], "tier": u["tier"],
                "repairability": u["repairability"],
                "unseen_prompt": u["unseen_prompt"],
                "ref": u["ref"], "a0": u["a0"], "d1_text": picked_text,
                "picked_idx": pick, "label_idx": u["label"],
                "p_a0": float(probs[0]) if u["texts"][0] == u["a0"] else None,
                "p_picked": float(probs[pick]),
                "kept_a0": picked_text == u["a0"],
                "a0_wer": u["a0_wer"],
                "oracle_wer": u["oracle_wer"],
            })
        fold_stats.append({"fold": int(fold),
                           "test_speaker": test_u[0]["speaker_id"],
                           "weights": {k: round(float(v), 4)
                                       for k, v in zip(FEATURES, w)}})

    df = pd.DataFrame(rows)
    df["d1_wer_row"] = [jiwer.wer(r, h if h.strip() else "*")
                        for r, h in zip(df["ref"], df["d1_text"])]
    df["outcome"] = ["IMPROVED" if a < b - 1e-9 else "WORSENED" if a > b + 1e-9
                     else "UNCHANGED"
                     for b, a in zip(df["a0_wer"], df["d1_wer_row"])]
    df.to_csv(OUT / "d1_predictions.csv", index=False)

    def corpus_wer(refs, hyps):
        return jiwer.process_words(list(refs),
                                   [h if str(h).strip() else "*" for h in hyps]).wer

    def block(g):
        n = len(g)
        a0w, d1w = corpus_wer(g["ref"], g["a0"]), corpus_wer(g["ref"], g["d1_text"])
        orw = corpus_wer(g["ref"], [by_id[s]["oracle_text"] for s in g["sample_id"]])
        switches = g[~g["kept_a0"]]
        correct = g[g["ref"] == g["a0"]]
        return {
            "n": n, "a0_wer": a0w, "d1_wer": d1w, "oracle_wer": orw,
            "oracle_regret": d1w - orw,
            "gain_capture": (a0w - d1w) / (a0w - orw) if a0w - orw > 1e-9 else None,
            "a0_exact": float((g["ref"] == g["a0"]).mean()),
            "d1_exact": float((g["ref"] == g["d1_text"]).mean()),
            "selector_label_accuracy": float((g["picked_idx"] == g["label_idx"]).mean()),
            "keep_a0_rate": float(g["kept_a0"].mean()),
            "switch_rate": float((~g["kept_a0"]).mean()),
            "improved": int((g["outcome"] == "IMPROVED").sum()),
            "unchanged": int((g["outcome"] == "UNCHANGED").sum()),
            "worsened": int((g["outcome"] == "WORSENED").sum()),
            "edit_precision": (float((switches["outcome"] == "IMPROVED").mean())
                               if len(switches) else None),
            "preservation": (float(correct["kept_a0"].mean()) if len(correct) else None),
            "unsupported_generation_rate": 0.0,  # asserted per row
        }

    agg = block(df)
    per_speaker = {s: block(g) for s, g in df.groupby("speaker_id")}
    report = {
        "model": "conditional logit, shared scorer, "
                 f"{len(FEATURES)} trainable parameters",
        "features": FEATURES,
        "aggregate": agg,
        "per_speaker": per_speaker,
        "per_tier": {t: block(g) for t, g in df.groupby("tier")},
        "unseen_prompts": block(df[df["unseen_prompt"]]),
        "low": block(df[df["repairability"] == "LOW"]),
        "fold_weights": fold_stats,
    }
    (OUT / "d1_evaluation.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"\n=== D1 AGGREGATE ({agg['n']} sentences) ===")
    print(f"A0 {agg['a0_wer']:.4f} -> D1 {agg['d1_wer']:.4f} | oracle {agg['oracle_wer']:.4f}")
    print(f"oracle regret {agg['oracle_regret']:.4f} | GAIN CAPTURE "
          f"{agg['gain_capture'] * 100:.1f}%")
    print(f"exact {agg['a0_exact']*100:.1f}% -> {agg['d1_exact']*100:.1f}% | "
          f"label accuracy {agg['selector_label_accuracy']*100:.1f}%")
    print(f"KEEP_A0 {agg['keep_a0_rate']*100:.1f}% | switch {agg['switch_rate']*100:.1f}%")
    print(f"improved {agg['improved']} / unchanged {agg['unchanged']} / "
          f"worsened {agg['worsened']} (ratio "
          f"{agg['improved']/max(agg['worsened'],1):.2f}) | "
          f"edit precision {agg['edit_precision']*100:.1f}%")
    print(f"preservation {agg['preservation']*100:.2f}% | unsupported generation 0%")
    print("\n=== PER SPEAKER ===")
    for s in sorted(per_speaker, key=lambda x: per_speaker[x]["a0_wer"]):
        b = per_speaker[s]
        gc = f"{b['gain_capture']*100:5.1f}%" if b["gain_capture"] is not None else "  n/a"
        print(f"  {s} n={b['n']:>3} A0 {b['a0_wer']:.4f} -> D1 {b['d1_wer']:.4f} "
              f"(oracle {b['oracle_wer']:.4f}, capture {gc}) "
              f"+{b['improved']}/-{b['worsened']} keep {b['keep_a0_rate']*100:.0f}%")
    for name in ("unseen_prompts", "low"):
        b = report[name]
        print(f"\n{name}: n={b['n']} A0 {b['a0_wer']:.4f} -> D1 {b['d1_wer']:.4f} "
              f"+{b['improved']}/-{b['worsened']} keep {b['keep_a0_rate']*100:.0f}%")


if __name__ == "__main__":
    main()
