"""Frozen N-best generation for the feasibility gate (dev speakers only).

    python scripts/nbest_generate.py

FROZEN configuration (see experiment2_nbest_design.md §4). Not tunable after
LOSO test evaluation begins:
  beam_size=12, num_hypotheses=8, keep top-5 unique after
  normalize -> loop-collapse -> dedupe; Silero VAD trim with faster-whisper
  defaults (production parity); beam scores cached, never prompted;
  loop-collapse is reference-independent.

Stage 1 (gate) covered the 6 development speakers. Stage 2 — authorized after
the gate passed and the hybrid-list amendment was approved — extends the SAME
frozen configuration to the held-out speakers (F04, M05) and the control set.
Nothing about the decoding is changed; the cache resumes incrementally.

Incremental cache: data/large_torgo/nbest/nbest_cache.jsonl
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import ctranslate2
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lt_audio import iter_audio  # noqa: E402
from utils import normalize_text  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / "data" / "large_torgo" / "nbest" / "nbest_cache.jsonl"

BEAM_SIZE = 12
NUM_HYPOTHESES = 8
KEEP = 5
DEV_SPEAKERS = ["F03", "M01", "M02", "M03", "F01", "M04"]


def collapse_loops(tokens: list[str]) -> list[str]:
    """Collapse a phrase of >=2 words repeated consecutively >=2 times.

    Reference-independent: uses only the hypothesis itself. Runs to fixpoint.
    """
    changed = True
    while changed:
        changed = False
        n = len(tokens)
        for period in range(min(n // 2, 12), 1, -1):
            i = 0
            while i + 2 * period <= len(tokens):
                if tokens[i:i + period] == tokens[i + period:i + 2 * period]:
                    del tokens[i + period:i + 2 * period]
                    changed = True
                else:
                    i += 1
    return tokens


def process_hypothesis(raw: str) -> str:
    return " ".join(collapse_loops(normalize_text(raw).split()))


def main() -> None:
    splits = pd.read_csv(PROJECT / "data/large_torgo/splits.csv",
                         keep_default_na=False, na_values=[""])
    work = splits[splits["split"].isin(["train", "validation", "test",
                                        "control_test"])]

    done = set()
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            try:
                done.add(json.loads(line)["sample_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    todo = [s for s in work["sample_id"] if s not in done]
    print(f"dev set: {len(work)} clips | cached: {len(done)} | to generate: {len(todo)}")
    if not todo:
        return

    from faster_whisper import WhisperModel
    from faster_whisper.vad import VadOptions, get_speech_timestamps, collect_chunks
    model = WhisperModel("medium.en", device="auto", compute_type="int8")
    fe = model.feature_extractor
    prompt = [model.hf_tokenizer.token_to_id(t)
              for t in ["<|startoftranscript|>", "<|notimestamps|>"]]
    vad_options = VadOptions()  # faster-whisper defaults = production vad_filter=True

    meta = work.set_index("sample_id")
    with CACHE.open("a") as out:
        for sample_id, raw in tqdm(iter_audio(todo), total=len(todo),
                                   unit="clip", dynamic_ncols=True):
            data, sr = sf.read(io.BytesIO(raw), dtype="float32")
            if sr != 16000:
                idx = np.linspace(0, len(data) - 1,
                                  int(len(data) * 16000 / sr)).astype(int)
                data = data[idx]
            # --- VAD trim (production parity) ------------------------------
            chunks = get_speech_timestamps(data, vad_options)
            trimmed = collect_chunks(data, chunks) if chunks else [data]
            audio = np.concatenate(trimmed) if isinstance(trimmed, list) else trimmed
            if len(audio) < 160:  # <10 ms of speech: fall back to raw audio
                audio = data

            feats = fe(audio, padding=True)[:, :3000]
            fv = ctranslate2.StorageView.from_array(
                np.ascontiguousarray(feats[None]).astype(np.float32))
            started = time.perf_counter()
            res = model.model.generate(fv, [prompt], beam_size=BEAM_SIZE,
                                       num_hypotheses=NUM_HYPOTHESES,
                                       return_scores=True, max_length=200)[0]
            elapsed = time.perf_counter() - started

            hyps = []
            for seq, score in zip(res.sequences_ids, res.scores):
                raw_text = model.hf_tokenizer.decode(seq).strip()
                hyps.append({"raw": raw_text,
                             "processed": process_hypothesis(raw_text),
                             "score": float(score)})
            seen, unique = set(), []
            for h in hyps:  # already ranked best-first by ct2
                if h["processed"] and h["processed"] not in seen:
                    seen.add(h["processed"])
                    unique.append(h)
                if len(unique) >= KEEP:
                    break

            record = {
                "sample_id": sample_id,
                "speaker_id": meta.loc[sample_id, "speaker_id"],
                "ground_truth": meta.loc[sample_id, "ground_truth"],
                "hypotheses": unique,          # top-5 unique, best first
                "a1_top1": unique[0]["processed"] if unique else "",
                "n_raw": len(hyps),
                "n_unique": len(unique),
                "generate_seconds": round(elapsed, 3),
            }
            out.write(json.dumps(record) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
