"""E01 -- candidate-pool expansion (Family A). FIXED configs, declared here.

Hypothesis (written before results): the H1-H5 oracle (0.2132) is limited by
list size and single-config decoding; a fixed union of (C1) wide beam N=24,
(C2) temperature sampling, and (C3) a second ASR model (large-v3-turbo) will
push the pool oracle materially below 0.21, possibly toward 0.15-0.17.

Configs are FIXED for all utterances; no reference-informed per-utterance
choices. Caches are per-config JSONL, resumable.
"""
import io, json, sys, time
from pathlib import Path
import numpy as np, soundfile as sf

RESEARCH = Path.home() / "Desktop/revoice-model-training"
sys.path.insert(0, str(RESEARCH / "scripts"))
from lt_audio import iter_audio            # noqa: E402
from utils import normalize_text           # noqa: E402
from nbest_generate import collapse_loops  # noqa: E402  (frozen loop-collapse)
import pandas as pd                        # noqa: E402

RW = Path.home() / "Downloads/revoice-overnight-research-20260830"
OUT = RW / "results"
CFGS = {
    # name: (model, decode kwargs for ct2 generate)
    "C1_medium_beam24": ("medium.en", dict(beam_size=24, num_hypotheses=24)),
    "C2_medium_sample": ("medium.en", dict(beam_size=1, num_hypotheses=12,
                                           sampling_temperature=0.6, sampling_topk=0)),
    "C3_turbo_beam12": ("large-v3-turbo", dict(beam_size=12, num_hypotheses=8)),
}

def clips():
    sp = pd.read_csv(RESEARCH / "data/large_torgo/splits.csv",
                     keep_default_na=False, na_values=[""])
    return sp[sp["split"].isin(["train", "validation", "test"])]["sample_id"].tolist()

def main(cfg_name):
    model_name, kwargs = CFGS[cfg_name]
    cache = OUT / f"pool_{cfg_name}.jsonl"
    done = set()
    if cache.exists():
        for l in cache.read_text().splitlines():
            if l.strip():
                done.add(json.loads(l)["sample_id"])
    ids = [c for c in clips() if c not in done]
    print(f"{cfg_name}: {len(done)} cached, {len(ids)} to decode", flush=True)
    if not ids:
        return
    import ctranslate2
    from faster_whisper import WhisperModel
    from faster_whisper.vad import VadOptions, get_speech_timestamps, collect_chunks
    m = WhisperModel(model_name, device="auto", compute_type="int8")
    if model_name.endswith(".en"):
        prompt = [m.hf_tokenizer.token_to_id(t) for t in
                  ["<|startoftranscript|>", "<|notimestamps|>"]]
    else:
        prompt = [m.hf_tokenizer.token_to_id(t) for t in
                  ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]]
    vad = VadOptions()
    t_start = time.time()
    with cache.open("a") as out:
        for n, (sid, raw) in enumerate(iter_audio(ids)):
            data, sr = sf.read(io.BytesIO(raw), dtype="float32")
            if sr != 16000:
                idx = np.linspace(0, len(data)-1, int(len(data)*16000/sr)).astype(int)
                data = data[idx]
            ch = get_speech_timestamps(data, vad)
            if ch:
                pieces = collect_chunks(data, ch)
                data2 = np.concatenate(pieces) if isinstance(pieces, list) else pieces
                if len(data2) >= 160:
                    data = data2
            feats = m.feature_extractor(data, padding=True)[:, :3000]
            fv = ctranslate2.StorageView.from_array(
                np.ascontiguousarray(feats[None]).astype(np.float32))
            t0 = time.perf_counter()
            res = m.model.generate(fv, [prompt], return_scores=True,
                                   max_length=200, **kwargs)[0]
            dt = time.perf_counter() - t0
            hyps, seen = [], set()
            for seq, score in zip(res.sequences_ids, res.scores):
                txt = m.hf_tokenizer.decode(seq).strip()
                norm = " ".join(collapse_loops(normalize_text(txt).split()))
                if norm and norm not in seen:
                    seen.add(norm)
                    hyps.append({"norm": norm, "score": float(score)})
            out.write(json.dumps({"sample_id": sid, "config": cfg_name,
                                  "hyps": hyps, "decode_s": round(dt, 3)}) + "\n")
            out.flush()
            if n % 50 == 0:
                el = time.time() - t_start
                print(f"  {n}/{len(ids)} ({el/60:.1f} min)", flush=True)
    print(f"{cfg_name} DONE in {(time.time()-t_start)/60:.1f} min", flush=True)

if __name__ == "__main__":
    for name in (sys.argv[1:] or list(CFGS)):
        main(name)
