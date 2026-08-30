"""E03: full fine-tune whisper-small.en on dysarthric train speakers.

Split (fixed, speaker-disjoint, pre-existing from experiment 1):
  train: F03, M01, M02, M03  (sentences + isolated words, headMic, REAL audio only)
  val:   F01, M04 sentences (early stopping / checkpoint pick)
  test:  F04, M05 sentences (decoded exactly once, at the very end)
"""
import io, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf, torch

RESEARCH = Path.home()/"Desktop/revoice-model-training"
sys.path.insert(0, str(RESEARCH/"scripts"))
from lt_audio import audio_bytes_for      # noqa: E402
from utils import normalize_text          # noqa: E402
import jiwer                              # noqa: E402

RW = Path.home()/"Downloads/revoice-overnight-research-20260830"
OUT = RW/"models/whisper_small_ft"; OUT.mkdir(parents=True, exist_ok=True)
TRAIN_SPK = ["F03","M01","M02","M03"]; VAL_SPK = ["F01","M04"]; TEST_SPK = ["F04","M05"]
SEED = 20260830; EPOCHS = 3; BS = 8; LR = 1e-5

from transformers import WhisperForConditionalGeneration, WhisperProcessor
torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en").to(device)
model.generation_config.language = None  # english-only checkpoint

meta = pd.read_csv(RESEARCH/"data/large_torgo/metadata.csv", keep_default_na=False, na_values=[""])
hm = meta[(meta.microphone=="headMic") & (meta.speaker_group=="dysarthric")]
train_rows = hm[hm.speaker_id.isin(TRAIN_SPK)][["sample_id","ground_truth","speaker_id","utterance_type"]]
val_rows   = hm[(hm.speaker_id.isin(VAL_SPK)) & (hm.utterance_type=="sentence")][["sample_id","ground_truth","speaker_id"]]
print(f"train {len(train_rows)} clips (incl words), val {len(val_rows)} sentences", flush=True)

def load_audio(sids):
    got = audio_bytes_for(list(sids))
    out = {}
    for sid, raw in got.items():
        d, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if sr != 16000:
            idx = np.linspace(0, len(d)-1, int(len(d)*16000/sr)).astype(int); d = d[idx]
        out[sid] = d
    return out

print("loading train audio into RAM...", flush=True)
train_audio = load_audio(train_rows.sample_id)   # ~2.5h float32 ≈ 1.1GB, fits
val_audio = load_audio(val_rows.sample_id)

def batches(rows, bs, shuffle=True):
    idx = np.random.permutation(len(rows)) if shuffle else np.arange(len(rows))
    for i in range(0, len(idx), bs):
        yield rows.iloc[idx[i:i+bs]]

def collate(rows, audio_map):
    feats = processor.feature_extractor([audio_map[s] for s in rows.sample_id],
                                        sampling_rate=16000, return_tensors="pt")
    labels = processor.tokenizer([normalize_text(t) for t in rows.ground_truth],
                                 return_tensors="pt", padding=True).input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return feats.input_features, labels

@torch.no_grad()
def val_wer():
    model.eval(); hyps, refs = [], []
    for rows in batches(val_rows, 8, shuffle=False):
        feats, _ = collate(rows, val_audio)
        gen = model.generate(feats.to(device), max_length=128, num_beams=2)
        hyps += [normalize_text(t) for t in processor.batch_decode(gen, skip_special_tokens=True)]
        refs += [normalize_text(t) for t in rows.ground_truth]
    return jiwer.process_words(refs, [h or "*" for h in hyps]).wer

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
print(f"baseline (unadapted whisper-small.en) val WER: {val_wer():.4f}", flush=True)
best = 1e9
t0 = time.time()
for ep in range(1, EPOCHS+1):
    model.train(); losses = []
    for n, rows in enumerate(batches(train_rows, BS)):
        feats, labels = collate(rows, train_audio)
        loss = model(input_features=feats.to(device), labels=labels.to(device)).loss
        loss.backward(); opt.step(); opt.zero_grad()
        losses.append(loss.item())
        if n % 40 == 0:
            print(f"  ep{ep} step {n}/{len(train_rows)//BS} loss={np.mean(losses[-40:]):.3f} ({(time.time()-t0)/60:.0f}m)", flush=True)
    vw = val_wer()
    print(f"epoch {ep}: mean loss {np.mean(losses):.3f} | val WER {vw:.4f} | {(time.time()-t0)/60:.0f} min", flush=True)
    if vw < best:
        best = vw
        model.save_pretrained(OUT); processor.save_pretrained(OUT)
        print(f"  checkpoint saved (val {vw:.4f})", flush=True)
json.dump({"best_val_wer": best, "epochs": EPOCHS,
           "train_clips": len(train_rows), "minutes": (time.time()-t0)/60},
          open(OUT/"train_summary.json","w"), indent=1)
print("E03 TRAINING DONE", flush=True)
