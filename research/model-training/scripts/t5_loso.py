"""Experiment 2 -- leave-one-speaker-out rerun of the T5 repair baseline.

    python scripts/t5_loso.py

Controlled factors (fixed): medium.en ASR (cached, nothing retranscribed),
t5-small, 5 epochs, HIGH/MEDIUM x2 oversampling, headMic only, LOW excluded
from training but kept in evaluation.

8 folds from data/large_torgo/loso_folds.csv. Per fold: train on 6 speakers,
select the checkpoint on the difficulty-neighbour validation speaker, evaluate
on the held-out test speaker plus the fixed 194-sentence control set. Test rows
are never used for selection.

Writes results/t5_loso/{loso_predictions.csv,control_predictions.csv,
loso_results.json,loso_report.md}.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import normalize_text  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
LT_DATA = PROJECT / "data" / "large_torgo"
OUT = PROJECT / "results" / "t5_loso"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "google-t5/t5-small"
PREFIX = "repair speech: "
LEARNING_RATE = 5e-5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
MAX_LENGTH = 128
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.1
NUM_BEAMS = 4
EPOCHS = 5
ERROR_OVERSAMPLE = 2  # HIGH/MEDIUM rows appear twice per epoch; CORRECT once
SEED = 20260829

# Difficulty tiers by measured medium.en WER (easy <0.1, medium 0.1-0.35).
TIERS = {"M03": "easy", "F04": "easy", "F03": "medium", "M05": "medium",
         "M02": "hard", "F01": "hard", "M01": "hard", "M04": "hard"}


def device_() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class Pairs(Dataset):
    def __init__(self, inputs, targets):
        self.inputs, self.targets = list(inputs), list(targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return {"input": PREFIX + self.inputs[i], "target": self.targets[i]}


def collate(tokenizer):
    def fn(batch):
        enc = tokenizer([b["input"] for b in batch], max_length=MAX_LENGTH,
                        padding=True, truncation=True, return_tensors="pt")
        labels = tokenizer([b["target"] for b in batch], max_length=MAX_LENGTH,
                           padding=True, truncation=True, return_tensors="pt").input_ids
        labels[labels == tokenizer.pad_token_id] = -100
        enc["labels"] = labels
        return enc
    return fn


@torch.no_grad()
def generate(model, tokenizer, texts, device):
    model.eval()
    out = []
    for i in range(0, len(texts), EVAL_BATCH_SIZE):
        batch = [PREFIX + t for t in texts[i:i + EVAL_BATCH_SIZE]]
        enc = tokenizer(batch, max_length=MAX_LENGTH, padding=True,
                        truncation=True, return_tensors="pt").to(device)
        gen = model.generate(**enc, max_length=MAX_LENGTH, num_beams=NUM_BEAMS,
                             early_stopping=True)
        out.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return [normalize_text(t) for t in out]


def corpus_wer(refs, hyps):
    if not refs:
        return float("nan")
    return jiwer.process_words(refs, [h if h.strip() else "*" for h in hyps]).wer


def row_wer(ref, hyp):
    if not ref.strip():
        return float("nan")
    return jiwer.wer(ref, hyp if hyp.strip() else "*")


def run_fold(fold, fold_rows, pairs, control, tokenizer, device):
    train_ids = fold_rows[fold_rows["role"] == "train"]["sample_id"]
    val_ids = fold_rows[fold_rows["role"] == "validation"]["sample_id"]
    test_ids = fold_rows[fold_rows["role"] == "test"]["sample_id"]
    unseen = set(fold_rows[(fold_rows["role"] == "test")
                           & (fold_rows["unseen_prompt"])]["sample_id"])

    train = pairs[pairs["sample_id"].isin(train_ids)]
    eligible = train[train["repairability"].isin(["CORRECT", "HIGH", "MEDIUM"])]
    errors = eligible[eligible["repairability"].isin(["HIGH", "MEDIUM"])]
    # x2 oversampling: append one extra copy of every HIGH/MEDIUM row.
    boosted = pd.concat([eligible] + [errors] * (ERROR_OVERSAMPLE - 1), ignore_index=True)

    val = pairs[pairs["sample_id"].isin(val_ids)]
    test = pairs[pairs["sample_id"].isin(test_ids)].copy()

    torch.manual_seed(SEED + fold)
    np.random.seed(SEED + fold)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    loader = DataLoader(Pairs(boosted["repair_input"], boosted["repair_target"]),
                        batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                        collate_fn=collate(tokenizer))
    steps = len(loader) * EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    sched = get_linear_schedule_with_warmup(opt, int(steps * WARMUP_RATIO), steps)

    val_in, val_tg = list(val["repair_input"]), list(val["repair_target"])
    best_wer, best_state, best_epoch, history = float("inf"), None, None, []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            opt.step(); sched.step(); opt.zero_grad()
            losses.append(loss.item())
        vw = corpus_wer(val_tg, generate(model, tokenizer, val_in, device))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_wer": vw})
        print(f"  fold {fold} epoch {epoch}: loss={np.mean(losses):.4f} val_WER={vw:.4f}",
              flush=True)
        if vw < best_wer - 1e-6:
            best_wer, best_epoch = vw, epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- held-out test speaker (all rows, LOW included) --------------------
    test["t5_normalized"] = generate(model, tokenizer, list(test["repair_input"]), device)
    test["asr_wer_row"] = [row_wer(r, h) for r, h in
                           zip(test["ground_truth_normalized"],
                               test["asr_transcript_normalized"])]
    test["t5_wer_row"] = [row_wer(r, h) for r, h in
                          zip(test["ground_truth_normalized"], test["t5_normalized"])]
    test["outcome"] = ["IMPROVED" if a < b - 1e-9 else "WORSENED" if a > b + 1e-9
                       else "UNCHANGED"
                       for b, a in zip(test["asr_wer_row"], test["t5_wer_row"])]
    test["t5_edited"] = test["t5_normalized"] != test["asr_transcript_normalized"]
    test["unseen_prompt"] = test["sample_id"].isin(unseen)
    test["fold"] = fold

    # --- fixed control set, this fold's model ------------------------------
    ctl = control.copy()
    ctl["t5_normalized"] = generate(model, tokenizer, list(ctl["repair_input"]), device)
    ctl["asr_wer_row"] = [row_wer(r, h) for r, h in
                          zip(ctl["ground_truth_normalized"],
                              ctl["asr_transcript_normalized"])]
    ctl["t5_wer_row"] = [row_wer(r, h) for r, h in
                         zip(ctl["ground_truth_normalized"], ctl["t5_normalized"])]
    ctl["t5_edited"] = ctl["t5_normalized"] != ctl["asr_transcript_normalized"]
    ctl["fold"] = fold

    del model, best_state
    meta = {"fold": fold,
            "test_speaker": test["speaker_id"].iloc[0],
            "validation_speaker": val["speaker_id"].iloc[0],
            "best_epoch": best_epoch, "best_val_wer": best_wer,
            "train_rows_effective": int(len(boosted)),
            "train_correct": int((eligible["repairability"] == "CORRECT").sum()),
            "train_errors_before_oversample": int(len(errors)),
            "history": history}
    return test, ctl, meta


def main() -> None:
    started = time.time()
    pairs = pd.read_csv(LT_DATA / "repair_pairs.csv", keep_default_na=False, na_values=[""])
    for c in ("repair_input", "repair_target", "asr_transcript_normalized",
              "ground_truth_normalized"):
        pairs[c] = pairs[c].fillna("")
    folds = pd.read_csv(LT_DATA / "loso_folds.csv")
    control = pairs[pairs["split"] == "control_test"].reset_index(drop=True)

    device = device_()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"device: {device} | folds: {folds['fold'].nunique()} | "
          f"control rows per fold: {len(control)}", flush=True)

    tests, controls, metas = [], [], []
    for fold in sorted(folds["fold"].unique()):
        print(f"\n=== FOLD {fold} ===", flush=True)
        t, c, m = run_fold(fold, folds[folds["fold"] == fold], pairs,
                           control, tokenizer, device)
        tests.append(t); controls.append(c); metas.append(m)
        print(f"  test {m['test_speaker']}: ASR WER "
              f"{corpus_wer(list(t['ground_truth_normalized']), list(t['asr_transcript_normalized'])):.4f} "
              f"-> T5 {corpus_wer(list(t['ground_truth_normalized']), list(t['t5_normalized'])):.4f}",
              flush=True)

    all_test = pd.concat(tests, ignore_index=True)
    all_ctl = pd.concat(controls, ignore_index=True)
    all_test["tier"] = all_test["speaker_id"].map(TIERS)
    all_test.to_csv(OUT / "loso_predictions.csv", index=False)
    all_ctl.to_csv(OUT / "control_predictions.csv", index=False)
    (OUT / "loso_results.json").write_text(json.dumps(
        {"folds": metas, "elapsed_seconds": round(time.time() - started, 1)},
        indent=2, default=str))
    print(f"\nAll folds done in {(time.time() - started) / 60:.1f} min. "
          f"Wrote {OUT}/", flush=True)


if __name__ == "__main__":
    main()
