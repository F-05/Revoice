"""Families C+D: GPT-2 NLL + wav2vec2-CTC NLL for every unique candidate.

Scores are reference-independent and cached per (sid, candidate)."""
import io, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf, torch
RESEARCH = Path.home()/"Desktop/revoice-model-training"
sys.path.insert(0, str(RESEARCH/"scripts"))
from lt_audio import audio_bytes_for  # noqa: E402
RW = Path.home()/"Downloads/revoice-overnight-research-20260830"

pairs = pd.read_csv(RESEARCH/"data/large_torgo/repair_pairs.csv", keep_default_na=False, na_values=[""])
pairs = pairs[pairs.speaker_group=="dysarthric"].set_index("sample_id")

def load_all_pools():
    pools = {}
    for l in (RESEARCH/"data/large_torgo/nbest/nbest_cache.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l); pools.setdefault(r["sample_id"], set()).update(h["processed"] for h in r["hypotheses"])
    for name in ("C1_medium_beam24","C2_medium_sample","C3_turbo_beam12","C4_adapted_small"):
        p = RW/f"results/pool_{name}.jsonl"
        if p.exists():
            for l in p.read_text().splitlines():
                if l.strip():
                    r = json.loads(l); pools.setdefault(r["sample_id"], set()).update(h["norm"] for h in r["hyps"])
    for sid, row in pairs.iterrows():
        a0 = row["asr_transcript_normalized"]
        if a0: pools.setdefault(sid, set()).add(a0)
    return {k: sorted(x for x in v if x) for k, v in pools.items()}

def main(kind):
    cache = RW/f"results/scores_{kind}.jsonl"
    have = {}
    if cache.exists():
        for l in cache.read_text().splitlines():
            if l.strip():
                r = json.loads(l); have.setdefault(r["sample_id"], {}).update(r["scores"])
    pools = load_all_pools()
    todo = [s for s in pools if set(pools[s]) - set(have.get(s, {}))]
    print(f"{kind}: {len(have)} sids cached, {len(todo)} needing new candidates", flush=True)
    device = torch.device("cpu")  # MPS reserved for E03 training
    if kind == "gpt2":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        lm = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
        with cache.open("a") as out, torch.no_grad():
            for n, sid in enumerate(todo):
                scores = dict(have.get(sid, {}))
                for c in pools[sid]:
                    if c in scores: continue
                    ids = tok(c, return_tensors="pt").input_ids.to(device)
                    if ids.shape[1] < 2: scores[c] = 20.0; continue
                    loss = lm(ids, labels=ids).loss.item()
                    scores[c] = round(loss, 4)  # per-token NLL
                out.write(json.dumps({"sample_id": sid, "scores": scores})+"\n"); out.flush()
                if n % 50 == 0: print(f"  {n}/{len(todo)}", flush=True)
    else:  # w2v CTC
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        am = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()
        with cache.open("a") as out, torch.no_grad():
            for n, sid in enumerate(todo):
                raw = audio_bytes_for([sid])[sid]
                d, sr = sf.read(io.BytesIO(raw), dtype="float32")
                if sr != 16000:
                    idx = np.linspace(0, len(d)-1, int(len(d)*16000/sr)).astype(int); d = d[idx]
                feats = proc(d, sampling_rate=16000, return_tensors="pt").input_values.to(device)
                logits = am(feats).logits.log_softmax(-1)
                scores = dict(have.get(sid, {}))
                for c in pools[sid]:
                    if c in scores: continue
                    tgt = proc.tokenizer(c.upper(), return_tensors="pt").input_ids
                    if tgt.shape[1] == 0: scores[c] = 50.0; continue
                    nll = torch.nn.functional.ctc_loss(
                        logits.transpose(0,1), tgt.to(device),
                        torch.tensor([logits.shape[1]]), torch.tensor([tgt.shape[1]]),
                        blank=0, reduction="mean", zero_infinity=True).item()
                    scores[c] = round(nll, 4)
                out.write(json.dumps({"sample_id": sid, "scores": scores})+"\n"); out.flush()
                if n % 50 == 0: print(f"  {n}/{len(todo)}", flush=True)
    print(f"{kind} DONE", flush=True)

if __name__ == "__main__":
    main(sys.argv[1])
