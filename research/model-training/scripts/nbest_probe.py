"""Micro-probe: can ctranslate2 give REAL N-best lists for TORGO clips?

12 clips from train/validation speakers only. Not the full N-best generation --
just a mechanism + diversity check for the experiment-2 design doc.
"""
import sys
from pathlib import Path

import ctranslate2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lt_audio import audio_bytes_for
from utils import normalize_text
import jiwer
import pandas as pd
import io
import soundfile as sf

PROJECT = Path(__file__).resolve().parent.parent
splits = pd.read_csv(PROJECT / "data/large_torgo/splits.csv",
                     keep_default_na=False, na_values=[""])
# train+validation speakers only; mix of difficulties
pool = splits[splits["split"].isin(["train", "validation"])]
picked = pd.concat([g.sample(2, random_state=7) for _, g in pool.groupby("speaker_id")])
print(f"{len(picked)} clips from speakers {sorted(picked.speaker_id.unique())}")

from faster_whisper import WhisperModel
m = WhisperModel("medium.en", device="auto", compute_type="int8")
fe = m.feature_extractor

BEAM, N = 10, 5
rows = []
audio_map = audio_bytes_for(list(picked["sample_id"]))
for _, r in picked.iterrows():
    data, sr = sf.read(io.BytesIO(audio_map[r["sample_id"]]), dtype="float32")
    if sr != 16000:
        import math
        idx = np.linspace(0, len(data) - 1, int(len(data) * 16000 / sr)).astype(int)
        data = data[idx]
    feats = fe(data, padding=True)[:, :3000]
    fv = ctranslate2.StorageView.from_array(np.ascontiguousarray(feats[None]).astype(np.float32))
    prompt = [m.hf_tokenizer.token_to_id(t) for t in
              ["<|startoftranscript|>", "<|notimestamps|>"]]
    res = m.model.generate(fv, [prompt], beam_size=BEAM, num_hypotheses=N,
                           return_scores=True, max_length=200)[0]
    hyps = []
    for seq, score in zip(res.sequences_ids, res.scores):
        text = m.hf_tokenizer.decode(seq).strip()
        hyps.append((normalize_text(text), score))
    uniq = list(dict.fromkeys(h for h, _ in hyps if h))
    truth = normalize_text(r["ground_truth"])
    wers = [jiwer.wer(truth, h if h else "*") for h in uniq]
    rows.append({"speaker": r["speaker_id"], "truth": truth,
                 "n_unique": len(uniq), "hyps": uniq,
                 "best_wer": wers[0] if wers else None,
                 "oracle_wer": min(wers) if wers else None})
    print(f"\n[{r['speaker_id']}] truth: {truth!r}")
    for i, (h, s) in enumerate(zip(uniq, [w for w in wers])):
        print(f"   {i+1}. ({s:.2f}) {h!r}")

df = pd.DataFrame(rows)
print("\n=== PROBE SUMMARY (12 clips, beam=10, N=5) ===")
print(f"clips with >1 unique hypothesis: {(df.n_unique > 1).sum()}/{len(df)}")
print(f"mean unique hypotheses: {df.n_unique.mean():.2f}")
print(f"mean 1-best WER:  {df.best_wer.mean():.4f}")
print(f"mean oracle WER:  {df.oracle_wer.mean():.4f}")
print(f"oracle better on: {(df.oracle_wer < df.best_wer).sum()}/{len(df)} clips")
