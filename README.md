# Revoice

**Your voice, made clear.**

Revoice is an accessibility-focused speech clarification app for people whose
speech may be difficult for conventional speech recognition — or listeners —
to understand. It transcribes speech locally, conservatively repairs likely
recognition errors, and speaks the clarified sentence aloud. Revoice tries to
preserve what the speaker actually meant: it never freely rewrites speech, and
uncertain corrections are shown as suggestions the user confirms with one tap.
The goal is clearer communication without changing the speaker's voice or
meaning.

## Problem

Conventional ASR can struggle with atypical speech, including dysarthric
speech. A wrong transcription makes communication harder — especially when the
burden falls on the speaker to repeat themselves again and again. Revoice
explores whether *multiple* ASR hypotheses of the same utterance can be used
to recover a clearer interpretation, while remaining strictly conservative
about changing someone's words. Speech varies enormously between individuals;
Revoice does not assume any two speakers sound alike.

## How it works

```
Audio
  ↓
Whisper medium.en
  ↓
A0 + alternative hypotheses
  ↓
D3-NL constrained selector
  ↓
KEEP_A0 or UNCERTAIN
  ↓
"I think you said..."
  ↓
user confirmation
  ↓
TTS
```

- Whisper medium.en (faster-whisper / CTranslate2, fully local) produces the
  baseline transcript **A0**.
- One extra CTranslate2 beam pass yields alternative hypotheses; Revoice
  builds the hybrid list **H1–H5** (H1 = A0, H2–H5 = unique alternatives).
  All text canonicalization happens *before* this list is finalized.
- The **D3-NL selector** — a tiny neural ranker over **21 inference-safe
  features** (hypothesis consensus, decoder evidence, edit structure) — ranks
  the candidates. It ships as plain JSON weights and runs in NumPy.
- The deployed product is **suggestion-first**: if A0 is preferred, it is
  kept and the normal confidence flow applies. If another candidate is
  preferred, Revoice returns **UNCERTAIN** — the transcript stays A0, and the
  preferred hypothesis is shown as *"I think you said…"* for confirmation.
- The suggested text must be one of the actual ASR hypotheses.

**Revoice does not freely generate a replacement sentence in the deployed
pipeline.**

## Safety design

- **Constrained selection:** output is always a member of H1–H5; runtime
  invariants enforce it and fall back to A0 on any violation.
- **No post-selection rewriting:** candidate strings are final the moment the
  list is built.
- `suggested_text`, when present, is exactly `alternatives[0]`.
- **Automatic switching is disabled** for unknown/new speakers — suggestions
  always require confirmation and are never auto-spoken.
- **Fail closed:** a missing/invalid model artifact, N-best failure or
  selector exception never blocks transcription — the app degrades to plain
  Whisper.
- `REPAIR_BACKEND=none` is a one-line rollback to pre-selector behaviour.

## Results

Research evaluation: 683 dysarthric TORGO sentences, 8-fold
leave-one-speaker-out (every test speaker unseen by its fold's model).

| Metric | Result |
| --- | ---: |
| Whisper baseline WER | 31.75% |
| Revoice D2 WER | 28.49% |
| Relative error reduction | 10.3% |
| Helpful : harmful edits | 4.7 : 1 |
| Correct-input preservation | 98.91% |
| Control preservation | 99.92% |
| Unsupported generation | 0% |

WER = Word Error Rate; lower is better. 31.75% → 28.49% is a 3.26
percentage-point absolute reduction, i.e. about 10.3% of Whisper's word errors
removed — while leaving 98.91% of already-correct sentences untouched.

**Research headroom:** in development experiments, expanded candidate pools
contained a hypothesis at roughly **15–16% oracle WER**. That is *not*
achieved system performance — it means a much better candidate often already
exists in the pool; selecting it reliably remains future work.

### Research vs. product

| | |
| --- | --- |
| **Research result** | D2, frozen LOSO evaluation — 28.49% WER (table above) |
| **Deployed product** | D3-NL suggestion-first selector (chosen for its stronger safety profile) |
| **Development-only** | D4-dev — 27.60% WER, not deployed |
| **D5** | no meaningful improvement; not deployed |

## Tech stack

**Frontend:** React Native, Expo, TypeScript
**Backend:** Python, FastAPI, NumPy, faster-whisper, CTranslate2
**Model / research:** Whisper medium.en, constrained N-best selection, TORGO,
speaker-disjoint LOSO evaluation; research/training code additionally uses
PyTorch, Hugging Face Transformers and jiwer (research only — production
inference needs none of them).

## Project structure

```
revoice/
├── frontend/    Expo + React Native app (iPhone-first)
├── backend/     FastAPI, local Whisper, N-best generation, D3-NL selector
├── research/    model-training + experiment code, reports, small results
└── README.md
```

- `backend/app/assets/revoice_selector_v1.json` — the deployed selector
  artifact (validated at startup).
- `research/model-training/` — dataset preparation, LOSO evaluation, selector
  experiments. `research/overnight/` — candidate-generation and rescoring
  research, experiment registry and reports. Datasets and generated corpora
  are not committed.

## Run locally

Backend (Python 3.12):

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm install
cp .env.example .env
npx expo start --dev-client --lan
```

For a physical iPhone, `EXPO_PUBLIC_API_URL` must point at your computer's LAN
IP (e.g. `http://192.168.x.x:8000`) and the phone and computer must be on the
same network. `curl http://127.0.0.1:8000/health` should answer
`{"status":"ok"}`.

## Environment variables

Backend (`backend/.env`, see `.env.example`):

| variable | meaning |
| --- | --- |
| `WHISPER_MODEL=medium.en` | production ASR model — repair metrics assume it |
| `REPAIR_BACKEND=selector\|none` | `selector` = full pipeline; `none` = plain Whisper rollback |
| `REPAIR_MODEL_PATH` | path to `revoice_selector_v1.json` |
| `REPAIR_SWITCH_MARGIN` | optional operational/debug override of the suggestion-strength threshold; the deployed policy remains suggestion-first either way |

Frontend (`frontend/.env`): `EXPO_PUBLIC_API_URL` (backend address),
`EXPO_PUBLIC_DEMO_MODE` (mocked responses for UI demos).

## Demo flow

> **Whisper:** "We rode horse horse back to that farm."
>
> **Revoice suggestion:** "We rode horseback to that farm."
>
> *I think you said…* **[ Use this ]**

The app keeps the original transcript until the user confirms the suggestion;
only then is the clarified sentence spoken.

## Limitations

- The prototype is evaluated primarily on TORGO; its 8 dysarthric speakers
  were reused across development, so fresh-speaker validation remains future
  work.
- Selection thresholds do not transfer perfectly across speakers — one reason
  the deployed policy is suggestion-first rather than automatic.
- Suggestion-first is intentionally conservative: some good corrections are
  offered rather than applied.
- Latency is a few seconds per utterance, since multiple hypotheses are
  generated on-device/CPU.
- English-only for now; multilingual support is future work.

## What's next

- Speaker personalization (using confirmations to calibrate per-user behaviour)
- Stronger acoustic discrimination / hypothesis reranking
- Whisper adaptation / fine-tuning for atypical speech
- Multilingual support
- Broader evaluation on unseen speakers
- Phone / video-call integration
