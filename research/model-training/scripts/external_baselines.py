"""External baselines: GPT-4o / Gemini 2.5 Pro / Deepgram on the frozen set.

    python scripts/external_baselines.py --provider openai|gemini|deepgram
    python scripts/external_baselines.py --score

Evaluation set: the SAME frozen 683 headMic dysarthric sentences, same
references, same normalize_text, same jiwer scoring, same speaker/tier
breakdown as every Revoice experiment. Nothing from D2 is touched or tuned.

FROZEN INSTRUCTION (fixed before any API call; never varied):
  "Transcribe exactly what the speaker says. Do not paraphrase, correct
   grammar, or infer words that were not spoken."

Caching: one JSONL per provider under data/external_baselines/ — each clip is
sent to each API exactly once; reruns resume.

Required environment variables (STOP if absent — never substitute a model):
  openai   -> OPENAI_API_KEY      model: gpt-4o-transcribe
  gemini   -> GEMINI_API_KEY      model: gemini-2.5-pro
  deepgram -> DEEPGRAM_API_KEY    model: nova-3
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lt_audio import iter_audio  # noqa: E402
from utils import normalize_text  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
LT = PROJECT / "data" / "large_torgo"
CACHE_DIR = PROJECT / "data" / "external_baselines"
OUT = PROJECT / "results" / "external_baselines"
OUT.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = ("Transcribe exactly what the speaker says. Do not paraphrase, "
          "correct grammar, or infer words that were not spoken.")
TIERS = {"M03": "easy", "F04": "easy", "F03": "medium", "M05": "medium",
         "M02": "hard", "F01": "hard", "M01": "hard", "M04": "hard"}
MODELS = {"openai": "gpt-4o-transcribe", "gemini": "gemini-2.5-pro",
          "deepgram": "nova-3"}


def eval_set() -> pd.DataFrame:
    rows = pd.read_csv(PROJECT / "results/experiment3_selector/oracle_rows.csv",
                       keep_default_na=False, na_values=[""])
    return rows[["sample_id", "speaker_id", "ref"]]


def post(url, data, headers, timeout=120):
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body), time.perf_counter() - started


def call_openai(wav: bytes) -> tuple[str, float]:
    boundary = "----revoicebench"
    parts = []
    for name, value in (("model", MODELS["openai"]), ("prompt", PROMPT),
                        ("response_format", "json")):
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                     f"name=\"{name}\"\r\n\r\n{value}\r\n".encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                 f"name=\"file\"; filename=\"clip.wav\"\r\n"
                 f"Content-Type: audio/wav\r\n\r\n".encode() + wav + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    out, dt = post("https://api.openai.com/v1/audio/transcriptions", body,
                   {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                    "Content-Type": f"multipart/form-data; boundary={boundary}"})
    return out.get("text", ""), dt


def call_gemini(wav: bytes) -> tuple[str, float]:
    payload = json.dumps({
        "contents": [{"parts": [
            {"text": PROMPT},
            {"inline_data": {"mime_type": "audio/wav",
                             "data": base64.b64encode(wav).decode()}}]}],
        "generationConfig": {"temperature": 0},
    }).encode()
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{MODELS['gemini']}:generateContent?key={os.environ['GEMINI_API_KEY']}")
    out, dt = post(url, payload, {"Content-Type": "application/json"})
    try:
        text = out["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        text = ""
    return text, dt


def call_deepgram(wav: bytes) -> tuple[str, float]:
    out, dt = post(f"https://api.deepgram.com/v1/listen?model={MODELS['deepgram']}"
                   "&smart_format=false&punctuate=false", wav,
                   {"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}",
                    "Content-Type": "audio/wav"})
    try:
        text = out["results"]["channels"][0]["alternatives"][0]["transcript"]
    except (KeyError, IndexError):
        text = ""
    return text, dt


CALLERS = {"openai": call_openai, "gemini": call_gemini, "deepgram": call_deepgram}
KEYS = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY",
        "deepgram": "DEEPGRAM_API_KEY"}


def run(provider: str) -> None:
    if not os.environ.get(KEYS[provider]):
        sys.exit(f"STOP: {KEYS[provider]} is not set. Provide the credential; "
                 "no substitute model will be used.")
    cache = CACHE_DIR / f"{provider}.jsonl"
    done = set()
    if cache.exists():
        for line in cache.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["sample_id"])
    work = eval_set()
    todo = [s for s in work["sample_id"] if s not in done]
    print(f"{provider} ({MODELS[provider]}): {len(done)} cached, {len(todo)} to call")
    from tqdm import tqdm
    with cache.open("a") as out:
        for sid, wav in tqdm(iter_audio(todo), total=len(todo), unit="clip"):
            for attempt in range(3):
                try:
                    text, dt = CALLERS[provider](wav)
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 2:
                        text, dt = "", None
                        print(f"  {sid}: FAILED after 3 tries: {e}")
                    else:
                        time.sleep(5 * (attempt + 1))
            out.write(json.dumps({"sample_id": sid, "provider": provider,
                                  "model": MODELS[provider], "text": text,
                                  "latency_sec": dt,
                                  "date": time.strftime("%Y-%m-%d")}) + "\n")
            out.flush()


def score() -> None:
    import jiwer

    def corpus_wer(refs, hyps):
        return jiwer.process_words(list(refs),
                                   [h if str(h).strip() else "*" for h in hyps]).wer

    work = eval_set()
    report = {"frozen_comparison": {"A0_whisper_medium_en": 0.3175,
                                    "Revoice_D2": 0.2849},
              "prompt": PROMPT, "n_eval": len(work), "providers": {}}
    for provider in MODELS:
        cache = CACHE_DIR / f"{provider}.jsonl"
        if not cache.exists():
            continue
        recs = {r["sample_id"]: r for r in
                (json.loads(l) for l in cache.read_text().splitlines() if l.strip())}
        df = work[work["sample_id"].isin(recs)].copy()
        if df.empty:
            continue
        df["raw"] = [recs[s]["text"] for s in df["sample_id"]]
        df["hyp"] = df["raw"].map(normalize_text)
        df["tier"] = df["speaker_id"].map(TIERS)
        df["lat"] = [recs[s].get("latency_sec") for s in df["sample_id"]]
        # crude unsupported-output flag: normalized hyp > 3x reference length
        df["runaway"] = [len(h.split()) > 3 * max(len(r.split()), 1)
                         for h, r in zip(df["hyp"], df["ref"])]
        block = lambda g: {"n": len(g), "wer": corpus_wer(g["ref"], g["hyp"]),
                           "exact": float((g["ref"] == g["hyp"]).mean())}
        lat = df["lat"].dropna()
        report["providers"][provider] = {
            "model": MODELS[provider],
            "date_tested": recs[df["sample_id"].iloc[0]].get("date"),
            "coverage": len(df),
            "aggregate": block(df),
            "per_speaker": {s: block(g) for s, g in df.groupby("speaker_id")},
            "per_tier": {t: block(g) for t, g in df.groupby("tier")},
            "empty_rate": float((df["hyp"].str.strip() == "").mean()),
            "runaway_rate_gt3x": float(df["runaway"].mean()),
            "latency_mean_sec": float(lat.mean()) if len(lat) else None,
        }
        df.to_csv(OUT / f"{provider}_scored.csv", index=False)
    (OUT / "external_baselines.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)[:4000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=list(MODELS))
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.score:
        score()
    elif a.provider:
        run(a.provider)
    else:
        ap.error("need --provider or --score")
