"""Step 13 -- fine-tune google-t5/t5-small as the Revoice repair model.

    python scripts/t5_train.py

Task format:
    input   "repair speech: <asr transcript>"
    target  "<ground truth>"

Training data is the TRAIN speakers only, restricted to rows the triage marks
CORRECT / HIGH / MEDIUM. Checkpoints are selected on the VALIDATION speakers
(the full validation set, LOW-repairability rows included, because that is what
deployment actually faces). The test speakers are never read here.

Writes models/revoice-t5-small/ and results/t5_small/training_summary.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import normalize_text  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
LT_DATA = PROJECT / "data" / "large_torgo"
OUT_MODEL = PROJECT / "models" / "revoice-t5-small"
OUT_RESULTS = PROJECT / "results" / "t5_small"
OUT_MODEL.mkdir(parents=True, exist_ok=True)
OUT_RESULTS.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "google-t5/t5-small"
PREFIX = "repair speech: "

LEARNING_RATE = 5e-5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
MAX_LENGTH = 128
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.1
NUM_BEAMS = 4
SEED = 20260829

# The planned baseline is exactly 5 epochs. With ~415 training sentences over
# heavily repeated TORGO prompts, training longer risks memorising the prompt
# list rather than learning to repair, so the run is NOT extended on the basis
# of dataset size. Every epoch is logged; if validation metrics are still
# clearly improving at epoch 5 that justifies a separate longer run.
EPOCHS = 5


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RepairDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer):
        self.inputs = [PREFIX + t for t in frame["repair_input"].fillna("")]
        self.targets = list(frame["repair_target"].fillna(""))
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict:
        return {"input": self.inputs[index], "target": self.targets[index]}


def make_collate(tokenizer):
    def collate(batch: list[dict]) -> dict:
        model_inputs = tokenizer([b["input"] for b in batch], max_length=MAX_LENGTH,
                                 padding=True, truncation=True, return_tensors="pt")
        labels = tokenizer([b["target"] for b in batch], max_length=MAX_LENGTH,
                           padding=True, truncation=True, return_tensors="pt").input_ids
        # -100 keeps padding out of the loss.
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs
    return collate


@torch.no_grad()
def validation_loss(model, loader, device) -> float:
    """Teacher-forced loss on the validation speakers."""
    model.eval()
    losses = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        losses.append(model(**batch).loss.item())
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def generate(model, tokenizer, texts: list[str], device, batch_size: int = EVAL_BATCH_SIZE):
    model.eval()
    out: list[str] = []
    for i in range(0, len(texts), batch_size):
        batch = [PREFIX + t for t in texts[i:i + batch_size]]
        encoded = tokenizer(batch, max_length=MAX_LENGTH, padding=True,
                            truncation=True, return_tensors="pt").to(device)
        generated = model.generate(**encoded, max_length=MAX_LENGTH,
                                   num_beams=NUM_BEAMS, early_stopping=True)
        out.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return out


def corpus_wer(references: list[str], hypotheses: list[str]) -> float:
    return jiwer.process_words(references,
                               [h if h.strip() else "*" for h in hypotheses]).wer


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    pairs = pd.read_csv(LT_DATA / "repair_pairs.csv", keep_default_na=False, na_values=[""])
    for column in ("repair_input", "repair_target"):
        pairs[column] = pairs[column].fillna("")

    train = pairs[pairs["train_eligible"]].reset_index(drop=True)
    validation = pairs[pairs["split"] == "validation"].reset_index(drop=True)
    excluded = pairs[(pairs["split"] == "train") & (~pairs["train_eligible"])]

    print(f"train rows (eligible): {len(train)}")
    print(f"  composition: {train['repairability'].value_counts().to_dict()}")
    print(f"  excluded LOW-repairability train rows: {len(excluded)}")
    print(f"validation rows (all, LOW included): {len(validation)}")

    device = pick_device()
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

    loader = DataLoader(RepairDataset(train, tokenizer), batch_size=TRAIN_BATCH_SIZE,
                        shuffle=True, collate_fn=make_collate(tokenizer))
    total_steps = len(loader) * EPOCHS
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * WARMUP_RATIO), total_steps)

    val_loader = DataLoader(RepairDataset(validation, tokenizer), batch_size=EVAL_BATCH_SIZE,
                            shuffle=False, collate_fn=make_collate(tokenizer))
    val_inputs = list(validation["repair_input"])
    val_targets = list(validation["repair_target"])
    baseline_val_wer = corpus_wer(val_targets, val_inputs)
    baseline_val_exact = float(np.mean([i == t for i, t in zip(val_inputs, val_targets)]))
    # Rows the ASR already got right: the repair model should leave these alone.
    correct_idx = [i for i, (a, t) in enumerate(zip(val_inputs, val_targets)) if a == t]
    print(f"validation WER of raw ASR (the bar to beat): {baseline_val_wer:.4f}")
    print(f"validation exact match of raw ASR: {baseline_val_exact:.4f}")
    print(f"validation rows where ASR was already correct: {len(correct_idx)}")

    history, best = [], {"epoch": None, "val_wer": float("inf")}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        progress = tqdm(loader, desc=f"epoch {epoch}", unit="batch", dynamic_ncols=True)
        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            progress.set_postfix_str(f"loss={np.mean(losses[-20:]):.4f}")

        val_loss = validation_loss(model, val_loader, device)
        predictions = generate(model, tokenizer, val_inputs, device)
        predictions = [normalize_text(p) for p in predictions]
        val_wer = corpus_wer(val_targets, predictions)
        exact = float(np.mean([p == t for p, t in zip(predictions, val_targets)]))
        preserved = (float(np.mean([predictions[i] == val_inputs[i] for i in correct_idx]))
                     if correct_idx else None)

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "val_wer": val_wer,
            "val_exact_match": exact,
            "val_correct_input_preservation": preserved,
        })
        print(f"  epoch {epoch}: train_loss={np.mean(losses):.4f} val_loss={val_loss:.4f} "
              f"val_WER={val_wer:.4f} val_exact={exact:.3f} "
              f"preservation={preserved if preserved is None else round(preserved, 3)}")

        # Checkpoint selection is on validation only; test is never consulted.
        if val_wer < best["val_wer"] - 1e-6:
            best = {"epoch": epoch, "val_wer": val_wer, "val_exact_match": exact,
                    "val_loss": val_loss, "val_correct_input_preservation": preserved}
            model.save_pretrained(OUT_MODEL)
            tokenizer.save_pretrained(OUT_MODEL)
            print(f"  ... new best, checkpoint saved to {OUT_MODEL}")

    asr_model = pairs["asr_model"].dropna().iloc[0] if pairs["asr_model"].notna().any() else None
    summary = {
        "base_model": MODEL_NAME,
        "task_prefix": PREFIX,
        "device": str(device),
        "seed": SEED,
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "warmup_ratio": WARMUP_RATIO,
            "num_beams": NUM_BEAMS,
            "epochs": EPOCHS,
            "note": "fixed 5-epoch baseline; not extended on the basis of dataset size, "
                    "because repeated TORGO prompts make long training a memorisation "
                    "risk. Per-epoch metrics are logged so a longer run can be justified "
                    "separately if validation is still improving at epoch 5.",
        },
        "asr_model": asr_model,
        "data": {
            "train_rows_eligible": int(len(train)),
            "train_composition": train["repairability"].value_counts().to_dict(),
            "train_correct_asr": int(train["asr_correct"].sum()),
            "train_incorrect_asr": int((~train["asr_correct"]).sum()),
            "train_excluded_low_repairability": int(len(excluded)),
            "validation_rows": int(len(validation)),
            "train_speakers": sorted(train["speaker_id"].unique().tolist()),
            "validation_speakers": sorted(validation["speaker_id"].unique().tolist()),
        },
        "validation_baseline_wer_raw_asr": baseline_val_wer,
        "validation_baseline_exact_match_raw_asr": baseline_val_exact,
        "validation_rows_asr_already_correct": len(correct_idx),
        "still_improving_at_final_epoch": (
            len(history) >= 2
            and history[-1]["val_wer"] < history[-2]["val_wer"] - 1e-6),
        "history": history,
        "best_checkpoint": best,
        "model_dir": str(OUT_MODEL),
    }
    (OUT_RESULTS / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (OUT_MODEL / "revoice_training_metadata.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"\nBest epoch {best['epoch']} (val WER {best['val_wer']:.4f}) "
          f"vs raw ASR {baseline_val_wer:.4f}")
    if summary["still_improving_at_final_epoch"]:
        print("NOTE: validation WER was still improving at the final epoch — a longer "
              "run may be justified, but that is a separate experiment.")
    print(f"Wrote {OUT_RESULTS / 'training_summary.json'}")


if __name__ == "__main__":
    main()
