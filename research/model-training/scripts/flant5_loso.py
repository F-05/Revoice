"""Experiment 2 System C -- hybrid N-best -> Flan-T5-base, LOSO.

    python scripts/flant5_loso.py

Input per utterance (pre-training architecture amendment, approved 2026-08-29,
justified by the dev-only feasibility gate):

    The following is a n-best list of ASR hypotheses for the given audio file:
    1. <cached production medium.en 1-best (A0)>
    2..5. <top-4 unique hypotheses from the FROZEN ct2 decoder, deduped vs A0>
    The correct transcription is:

The ct2 decoding configuration is untouched; only prompt-list construction
changed (A0 prepended). Target = normalized ground truth.

Training per fold: CORRECT retained + HIGH/MEDIUM x2 oversampling, LOW
excluded (labels from the 1-best triage, unchanged). Checkpoint selected on
the fold's validation speaker. Test rows never influence selection.

Outputs are separate from every previous experiment: results/t5_nbest/.
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
NBEST_CACHE = LT_DATA / "nbest" / "nbest_cache.jsonl"
OUT = PROJECT / "results" / "t5_nbest"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "google/flan-t5-base"
PREFIX = "The following is a n-best list of ASR hypotheses for the given audio file:\n"
SUFFIX = "The correct transcription is:"
LEARNING_RATE = 5e-5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
MAX_INPUT = 256
MAX_TARGET = 128
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.1
NUM_BEAMS = 4
EPOCHS = 5
ERROR_OVERSAMPLE = 2
SEED = 20260829


def device_():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_hybrid_inputs() -> dict[str, str]:
    """sample_id -> serialized hybrid prompt body (numbered list)."""
    pairs = pd.read_csv(LT_DATA / "repair_pairs.csv", keep_default_na=False, na_values=[""])
    a0_map = dict(zip(pairs["sample_id"], pairs["asr_transcript_normalized"].fillna("")))
    out = {}
    for line in NBEST_CACHE.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sid = r["sample_id"]
        a0 = a0_map.get(sid, "")
        hyps = [a0] if a0 else []
        seen = {a0} if a0 else set()
        for h in r["hypotheses"]:
            t = h["processed"]
            if t and t not in seen:
                seen.add(t)
                hyps.append(t)
            if len(hyps) >= 5:
                break
        if not hyps:
            hyps = [""]
        body = "".join(f"{i + 1}. {h}\n" for i, h in enumerate(hyps))
        out[sid] = PREFIX + body + SUFFIX
    return out


class Pairs(Dataset):
    def __init__(self, inputs, targets):
        self.inputs, self.targets = list(inputs), list(targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return {"input": self.inputs[i], "target": self.targets[i]}


def collate(tok):
    def fn(batch):
        enc = tok([b["input"] for b in batch], max_length=MAX_INPUT,
                  padding=True, truncation=True, return_tensors="pt")
        labels = tok([b["target"] for b in batch], max_length=MAX_TARGET,
                     padding=True, truncation=True, return_tensors="pt").input_ids
        labels[labels == tok.pad_token_id] = -100
        enc["labels"] = labels
        return enc
    return fn


@torch.no_grad()
def generate(model, tok, prompts, device):
    model.eval()
    out = []
    for i in range(0, len(prompts), EVAL_BATCH_SIZE):
        enc = tok(prompts[i:i + EVAL_BATCH_SIZE], max_length=MAX_INPUT, padding=True,
                  truncation=True, return_tensors="pt").to(device)
        gen = model.generate(**enc, max_length=MAX_TARGET, num_beams=NUM_BEAMS,
                             early_stopping=True)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return [normalize_text(t) for t in out]


def corpus_wer(refs, hyps):
    if not len(refs):
        return float("nan")
    return jiwer.process_words(list(refs),
                               [h if h.strip() else "*" for h in hyps]).wer


def row_wer(ref, hyp):
    if not ref.strip():
        return float("nan")
    return jiwer.wer(ref, hyp if hyp.strip() else "*")


def main() -> None:
    started = time.time()
    pairs = pd.read_csv(LT_DATA / "repair_pairs.csv", keep_default_na=False, na_values=[""])
    for c in ("asr_transcript_normalized", "ground_truth_normalized"):
        pairs[c] = pairs[c].fillna("")
    folds = pd.read_csv(LT_DATA / "loso_folds.csv")
    prompts = load_hybrid_inputs()
    pairs = pairs[pairs["sample_id"].isin(prompts)].copy()
    pairs["prompt"] = pairs["sample_id"].map(prompts)
    pairs["target"] = pairs["ground_truth_normalized"]
    control = pairs[pairs["split"] == "control_test"].reset_index(drop=True)

    device = device_()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"device: {device} | prompts loaded: {len(prompts)} | control: {len(control)}",
          flush=True)

    tests, controls, metas = [], [], []
    for fold in sorted(folds["fold"].unique()):
        fr = folds[folds["fold"] == fold]
        train = pairs[pairs["sample_id"].isin(fr[fr["role"] == "train"]["sample_id"])]
        val = pairs[pairs["sample_id"].isin(fr[fr["role"] == "validation"]["sample_id"])]
        test = pairs[pairs["sample_id"].isin(fr[fr["role"] == "test"]["sample_id"])].copy()
        unseen = set(fr[(fr["role"] == "test") & (fr["unseen_prompt"])]["sample_id"])

        eligible = train[train["repairability"].isin(["CORRECT", "HIGH", "MEDIUM"])]
        errors = eligible[eligible["repairability"].isin(["HIGH", "MEDIUM"])]
        boosted = pd.concat([eligible] + [errors] * (ERROR_OVERSAMPLE - 1),
                            ignore_index=True)

        print(f"\n=== FOLD {fold} (test {test['speaker_id'].iloc[0]}) "
              f"train_eff={len(boosted)} ===", flush=True)
        torch.manual_seed(SEED + fold)
        np.random.seed(SEED + fold)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
        loader = DataLoader(Pairs(boosted["prompt"], boosted["target"]),
                            batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                            collate_fn=collate(tok))
        steps = len(loader) * EPOCHS
        opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                weight_decay=WEIGHT_DECAY)
        sched = get_linear_schedule_with_warmup(opt, int(steps * WARMUP_RATIO), steps)

        val_prompts, val_tg = list(val["prompt"]), list(val["target"])
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
            vw = corpus_wer(val_tg, generate(model, tok, val_prompts, device))
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                            "val_wer": vw})
            print(f"  epoch {epoch}: loss={np.mean(losses):.4f} val_WER={vw:.4f}",
                  flush=True)
            if vw < best_wer - 1e-6:
                best_wer, best_epoch = vw, epoch
                best_state = copy.deepcopy(model.state_dict())
        if best_state is not None:
            model.load_state_dict(best_state)

        test["t5_normalized"] = generate(model, tok, list(test["prompt"]), device)
        ctl = control.copy()
        ctl["t5_normalized"] = generate(model, tok, list(ctl["prompt"]), device)
        for frame in (test, ctl):
            frame["asr_wer_row"] = [row_wer(r, h) for r, h in
                                    zip(frame["ground_truth_normalized"],
                                        frame["asr_transcript_normalized"])]
            frame["t5_wer_row"] = [row_wer(r, h) for r, h in
                                   zip(frame["ground_truth_normalized"],
                                       frame["t5_normalized"])]
            frame["outcome"] = ["IMPROVED" if a < b - 1e-9 else
                                "WORSENED" if a > b + 1e-9 else "UNCHANGED"
                                for b, a in zip(frame["asr_wer_row"], frame["t5_wer_row"])]
            frame["t5_edited"] = (frame["t5_normalized"]
                                  != frame["asr_transcript_normalized"])
            frame["fold"] = fold
        test["unseen_prompt"] = test["sample_id"].isin(unseen)
        print(f"  test: ASR {corpus_wer(test['ground_truth_normalized'], test['asr_transcript_normalized']):.4f} "
              f"-> C {corpus_wer(test['ground_truth_normalized'], test['t5_normalized']):.4f}",
              flush=True)

        tests.append(test.drop(columns=["prompt"]))
        controls.append(ctl.drop(columns=["prompt"]))
        metas.append({"fold": int(fold), "test_speaker": test["speaker_id"].iloc[0],
                      "best_epoch": best_epoch, "best_val_wer": best_wer,
                      "train_rows_effective": int(len(boosted)),
                      "history": history})
        del model, best_state

    pd.concat(tests, ignore_index=True).to_csv(OUT / "loso_predictions.csv", index=False)
    pd.concat(controls, ignore_index=True).to_csv(OUT / "control_predictions.csv", index=False)
    (OUT / "loso_results.json").write_text(json.dumps(
        {"model": MODEL_NAME, "folds": metas,
         "elapsed_seconds": round(time.time() - started, 1)}, indent=2, default=str))
    print(f"\nAll folds done in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
