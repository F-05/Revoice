# revoice-model-training

Standalone dataset-building, ASR evaluation and (later) repair-model training
for **Revoice**.

## Relationship to the Revoice app

This project is **completely independent** of the Revoice application at
`~/Desktop/speech-repair/`. It has its own virtualenv, its own dependency pins,
and its own copy of the ASR settings.

- Nothing here imports from `~/Desktop/speech-repair/backend/`.
- Nothing here writes to, or runs, the Revoice app.
- Nothing here calls the Revoice HTTP API — the model is invoked in-process.

`scripts/asr_config.py` is a **hand-copied mirror** of the backend's
faster-whisper settings (`backend/app/config.py` + `backend/app/services/asr.py`)
so that evaluation matches production. There is no automatic link: **if the
backend's ASR settings change, update `scripts/asr_config.py` by hand.**

Mirrored settings, as of 2026-08-29:

| setting | value |
|---|---|
| engine | `faster-whisper` 1.2.1 (CTranslate2) |
| `model_size` | `base.en` |
| `device` | `auto` (CPU on Apple Silicon — CTranslate2 has no Metal backend) |
| `compute_type` | `int8` |
| `language` | `en` |
| `beam_size` | `5` |
| `vad_filter` | `True` |
| `condition_on_previous_text` | `False` |
| `word_timestamps` | `True` |

`asr_confidence` is computed with the same formula as the backend: the
duration-weighted mean of `exp(avg_logprob)` over segments. It is a rough
proxy, not a calibrated probability, and is copied from the engine — never
invented.

Any of these can be overridden per run with the same env vars the backend uses:

```bash
WHISPER_MODEL=small.en ./.venv/bin/python scripts/transcribe_torgo.py --overwrite
```

## Dataset

[`resproj007/torgo_dysarthric_male`](https://huggingface.co/datasets/resproj007/torgo_dysarthric_male)
— 770 utterances (700 `train` / 70 `test`), columns `audio` and `text` only.
Audio is 24 kHz mono 16-bit WAV.

**Known limitations — do not overstate results from this data:**

- **No speaker IDs.** The dataset has no speaker, subject, or session column,
  and the stored audio `path` is `None` for every row. Speaker identity is
  therefore unavailable and is never inferred. **No speaker-disjoint
  evaluation is possible**, so utterances from the same speaker are almost
  certainly on both sides of any split.
- The upstream `train`/`test` split is **not** a clean split and is not
  treated as a trustworthy evaluation split. Beyond the missing speaker IDs,
  the 770 rows contain only **335 unique prompt texts**, and **65 texts appear
  on both sides** of the split. All 770 utterances are transcribed and reported
  together; the `split` column is preserved in the CSV for reference only.
- This is a **male dysarthric subset**. It does not represent all of TORGO,
  and it does not represent dysarthric speech in general.
- Word and sentence results are always reported separately and never pooled
  into a single headline number.

## Setup

```bash
python3.12 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

Python 3.12 is required — `python3` on this machine is 3.14, which lacks wheels
for parts of the ASR stack. `/opt/anaconda3/bin/python3.12` works.

## Usage

```bash
./.venv/bin/python scripts/inspect_dataset.py                 # 1. what is in the data
./.venv/bin/python scripts/transcribe_torgo.py                # 2-4. classify + run ASR + build CSV
./.venv/bin/python scripts/evaluate_whisper.py                # 5-7. normalize + WER + error report
```

`transcribe_torgo.py` appends each finished utterance to
`data/processed/whisper_predictions.jsonl` and flushes immediately, so a crash
costs at most one clip; re-running resumes from the cache. `--overwrite` forces
a fresh run, `--csv-only` rebuilds the CSV from the cache, `--limit N` is a
smoke test.

## Utterance classification

`utterance_type` is derived from the ground truth only:

- exactly **one** lexical word after normalization → `"word"` (e.g. `hill`, `swarm`)
- otherwise → `"sentence"` (e.g. `the misguided souls have lost their way`)

Neither category is dropped. They are evaluated separately: the **sentence**
subset is the metric that matters for Revoice, because the repair model is
meant to exploit linguistic context that a single isolated word does not carry.

## Text normalization

`utils.normalize_text` — NFKC, lowercase, strip punctuation (keeping intra-word
apostrophes and hyphens so `don't` stays one token), collapse whitespace, trim.
Used **only** for metric computation. The CSV keeps the untouched
`ground_truth` and `whisper_transcript` alongside the `*_normalized` columns.

## Layout

```
data/raw/                                 (unused so far — HF cache lives in .hf_cache/)
data/processed/whisper_predictions.jsonl  incremental, crash-safe ASR cache
scripts/asr_config.py                     mirror of the backend's ASR settings
scripts/utils.py                          normalization + classification + paths
scripts/inspect_dataset.py                step 1
scripts/transcribe_torgo.py               steps 2-4
scripts/evaluate_whisper.py               steps 5-7
evaluation/data/torgo_whisper.csv         the generated dataset
evaluation/data/torgo_whisper_scored.csv  + per-utterance WER
evaluation/reports/baseline_report.md     readable metrics + error examples
results/dataset_inspection.json           raw findings from step 1
results/baseline_metrics.json             machine-readable metrics
models/                                   (empty — no model has been trained)
```

## Baseline result (base.en, 770 utterances)

| subset | n | WER | exact match |
|---|---|---|---|
| overall | 770 | 0.886 | 11.82% |
| isolated words | 587 | 1.152 | 13.80% |
| sentences | 183 | 0.767 | 5.46% |

Word WER exceeds 1.0 because Whisper inserts words: 106 of 587 single-word
references drew a multi-token hypothesis, and some are long hallucinations
(`relax` -> "We live, oh, we live. We live. ..."). See
`evaluation/reports/baseline_report.md`.

## Status

Dataset generation and **baseline evaluation only**. No repair model has been
trained; `models/` is empty by design.
