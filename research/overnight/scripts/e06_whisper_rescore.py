"""E06: teacher-forced whisper-medium.en NLL for every pool candidate."""
import io, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf, torch
RESEARCH = Path.home()/"Desktop/revoice-model-training"
sys.path.insert(0, str(RESEARCH/"scripts"))
from lt_audio import audio_bytes_for  # noqa: E402
RW = Path.home()/"Downloads/revoice-overnight-research-20260830"
CACHE = RW/"results/scores_wmed.jsonl"

sys.path.insert(0, str(RW/"scripts"))
from score_candidates import load_all_pools  # noqa: E402

def main():
    have = {}
    if CACHE.exists():
        for l in CACHE.read_text().splitlines():
            if l.strip():
                r = json.loads(l); have.setdefault(r["sample_id"], {}).update(r["scores"])
    pools = load_all_pools()
    todo = [s for s in pools if set(pools[s]) - set(have.get(s, {}))]
    print(f"wmed: {len(have)} cached sids, {len(todo)} to score", flush=True)
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    proc = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-medium.en", torch_dtype=torch.float16).to(device).eval()
    t0 = time.time()
    with CACHE.open("a") as out, torch.no_grad():
        for n, sid in enumerate(todo):
            raw = audio_bytes_for([sid])[sid]
            d, sr = sf.read(io.BytesIO(raw), dtype="float32")
            if sr != 16000:
                idx = np.linspace(0, len(d)-1, int(len(d)*16000/sr)).astype(int); d = d[idx]
            feats = proc(d, sampling_rate=16000, return_tensors="pt").input_features.to(device, torch.float16)
            enc = model.get_encoder()(feats).last_hidden_state
            scores = dict(have.get(sid, {}))
            for c in pools[sid]:
                if c in scores: continue
                ids = proc.tokenizer(c, return_tensors="pt").input_ids.to(device)
                out_ = model(encoder_outputs=(enc,), labels=ids)
                # loss = mean per-token NLL of the candidate under medium.en
                scores[c] = round(float(out_.loss), 4)
            out.write(json.dumps({"sample_id": sid, "scores": scores})+"\n"); out.flush()
            if n % 25 == 0:
                el = (time.time()-t0)/60
                print(f"  {n}/{len(todo)} ({el:.1f}m)", flush=True)
    print("wmed DONE", flush=True)

if __name__ == "__main__":
    main()
