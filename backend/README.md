# Backend

FastAPI service for the speech pipeline:

```
Expo audio -> POST /process-speech -> (optional ffmpeg normalise) -> local Whisper
           -> N-best hypotheses -> constrained selector (suggestion-first)
           -> status + suggestion -> system TTS -> JSON (+ /audio/*.wav)
```

Repair is the **constrained suggestion-first selector** described in
"Repair: constrained hypothesis selection" below (`REPAIR_BACKEND=selector`) —
`repaired_text` is always the verbatim ASR sentence, with the selector's
preferred alternative offered separately for one-tap confirmation. With
`REPAIR_BACKEND=none|passthrough` the pre-selector placeholder behaviour is
preserved unchanged. TTS is the OS voice.

---

## Setup

### 1. Create the Python environment

Python **3.12** is recommended (3.10-3.13 all work; 3.14 does not have wheels
for every dependency yet).

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements-dev.txt   # runtime deps + pytest
```

Runtime only: `pip install -r requirements.txt`.

### 3. System dependencies

**ffmpeg is optional.** The default ASR backend (`faster-whisper`) decodes
`.m4a`, `.webm`, `.ogg` and friends itself through PyAV, so the service runs
without it — verified on a machine with no ffmpeg installed.

Install it if you want the extra container coverage, or if you switch to
`ASR_BACKEND=openai-whisper` (that backend shells out to ffmpeg and **requires**
it):

```bash
brew install ffmpeg
```

When ffmpeg is present the service normalises every upload to 16 kHz mono WAV
before inference. Set `TRANSCODE_WITH_FFMPEG=false` to skip that step.

### 4. Configuration

```bash
cp .env.example .env    # optional; every value has a working default
```

The first run downloads the Whisper weights from Hugging Face (~145 MB for
`base.en`) into `~/.cache/huggingface`. Later runs are offline.

### 5. Start the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

`--host 0.0.0.0` matters: it is what makes the server reachable from your phone.
Startup blocks until the model is loaded (~15 s cold, ~1 s warm). Set
`PRELOAD_ASR_MODEL=false` to boot instantly and pay that cost on the first
request instead.

---

## Testing without the frontend

### 6. `/health`

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status": "ok"}
```

### 7. `/process-speech`

Make a test recording — no microphone needed on macOS:

```bash
./scripts/make_sample_audio.sh "could you get me some water" sample.wav
```

Send it with the helper script:

```bash
python scripts/send_audio.py sample.wav
```

or with curl:

```bash
curl -X POST http://127.0.0.1:8000/process-speech -F "audio=@sample.wav"
```

Real output from that file:

```json
{
  "status": "success",
  "raw_transcript": "Could you get me some water?",
  "repaired_text": "Could you get me some water?",
  "confidence": 0.76,
  "uncertain_words": [],
  "audio_url": "/audio/result-c20840fc1d8d.wav"
}
```

Fetch the synthesised reply with `curl -O http://127.0.0.1:8000/audio/result-<id>.wav`,
or set `TTS_BACKEND=none` if you do not want the server speaking at all.

**Swagger** is at <http://127.0.0.1:8000/docs> — the *Try it out* button on
`POST /process-speech` gives you a file picker.

**Automated tests** run against the mock ASR backend, so they need no model and
finish in well under a second:

```bash
pytest
```

---

## 8. Reaching the backend from a phone on the same Wi-Fi

`localhost` on the phone is the *phone*, not your Mac. You need the Mac's LAN
address.

1. Find the LAN IP:

   ```bash
   ipconfig getifaddr en0
   ```

   (Try `en1` if `en0` is empty — `en0` is usually Wi-Fi on a laptop.)

2. Start the server bound to all interfaces:

   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

3. Check it from the Mac first, using the LAN IP rather than `127.0.0.1`:

   ```bash
   curl http://<LAN_IP>:8000/health
   python scripts/send_audio.py sample.wav --url http://<LAN_IP>:8000
   ```

4. Open `http://<LAN_IP>:8000/health` in Safari **on the phone**. If you see
   `{"status":"ok"}`, the frontend will work too.

5. In the Expo app, point the API base URL at `http://<LAN_IP>:8000`.

Gotchas, in the order they usually bite:

- **Phone and Mac must be on the same network.** Guest/IoT SSIDs and "client
  isolation" on the router will block this silently.
- **macOS firewall.** System Settings → Network → Firewall. Either turn it off
  on a trusted network or allow incoming connections for the Python binary.
- **Cleartext HTTP.** A dev build of Expo Go allows `http://` to a LAN IP. A
  release build needs ATS (iOS) / `usesCleartextTraffic` (Android) configured,
  or a tunnel.
- **The IP changes** when you rejoin a network. Do not hard-code it in the app.
- CORS is `*` by default, which is what you want here. Tighten
  `CORS_ALLOW_ORIGINS` before anything goes public.

---

## API contract

Stable. Additive changes only — tell the frontend before anything else moves.

### `GET /health`

```json
{"status": "ok"}
```

### `POST /process-speech`

Request: `multipart/form-data`, single field **`audio`**.
Accepted containers: wav, mp3, m4a, mp4, aac, ogg, opus, webm, flac, caf, aiff,
3gp, amr. Max 25 MB (`MAX_UPLOAD_BYTES`).

Response `200`:

| field | type | now | later |
|---|---|---|---|
| `status` | `"success" \| "uncertain" \| "retry"` | from confidence thresholds | calibrated |
| `raw_transcript` | `string \| null` | verbatim ASR output; `null` on `retry` | — |
| `repaired_text` | `string \| null` | transcript, tidied only | conservative repair |
| `confidence` | `number \| null` | 0.0–1.0, duration-weighted | calibrated |
| `uncertain_words` | `UncertainWord[]` | at most one, on `uncertain` | real alternatives |
| `audio_url` | `string \| null` | `/audio/...wav` on `success` | a better voice |

`UncertainWord` is `{ position: number, original: string, options: string[] }`, where
`position` is a zero-based index into `raw_transcript.split()` — the frontend blanks
out that word and offers `options` as choices.

`status` is what the app switches on:

| status | when | the app |
|---|---|---|
| `success` | every word is confident | speaks the sentence straight away |
| `uncertain` | some word is below `UNCERTAIN_WORD_THRESHOLD` | asks about that word first |
| `retry` | nothing usable was heard, or the clip is under `MIN_AUDIO_SECONDS` | asks for another take |

`audio_url` is **relative on purpose** — the frontend resolves it against its own
API base URL, which differs per device (localhost, LAN IP, tunnel). Files are served
from `GET /audio/<name>.wav` and pruned oldest-first past `AUDIO_KEEP_FILES`.

Errors use a different shape, distinguished by HTTP status:

```json
{
  "status": "error",
  "error": { "code": "invalid_audio", "message": "...", "detail": null }
}
```

| status | code |
|---|---|
| 400 | `invalid_audio` — empty file or unsupported extension |
| 413 | `audio_too_large` |
| 422 | `validation_error` — the `audio` field is missing |
| 422 | `audio_decode_failed` — no readable audio in the container |
| 500 | `transcription_failed`, `internal_error` |
| 503 | `asr_unavailable` — model could not be loaded |

---

## Layout

```
app/
  main.py            FastAPI app, CORS, lifespan (model preload)
  config.py          Settings (env / .env)
  dependencies.py    process-wide ASR singleton
  errors.py          typed errors -> JSON error responses
  api/routes.py      /health, /process-speech -- no engine logic here
  models/schemas.py  the API contract
  services/
    audio.py         temp files, size limits, decode probe, ffmpeg normalise
    asr.py           ASRService + WhisperASRService / OpenAIWhisper / Mock
    uncertainty.py   word scores -> status + uncertain_words
    repair.py        RepairService -- passthrough placeholder, milestone 2
    tts.py           TTSService -- system voice placeholder, milestone 4
```

### Swapping the ASR engine

`app/api/routes.py` only knows the `ASRService` interface. To swap engines, add
a subclass to `app/services/asr.py` and a branch in `build_asr_service`.

Three implementations ship today, selected with `ASR_BACKEND`:

| value | notes |
|---|---|
| `faster-whisper` | default. CTranslate2 runtime, no ffmpeg needed, fast on CPU |
| `openai-whisper` | reference implementation. `pip install openai-whisper`, needs ffmpeg + torch |
| `mock` | fixed transcript, no model. For tests and frontend integration |

`ASR_BACKEND=mock` is the quickest way to unblock the frontend: the API behaves
identically and starts instantly.

### Model choice

`WHISPER_MODEL` accepts `tiny.en`, `base.en`, `small.en`, `medium.en`,
`large-v3`. `base.en` is the default. Expect to move up to `small.en` once we
start measuring accuracy on real dysarthric speech.

Apple Silicon runs on CPU — CTranslate2 has no Metal backend. `int8` is the
right `WHISPER_COMPUTE_TYPE` there.

---

## Milestones

- [x] **1. Vertical slice** — audio → FastAPI → Whisper → transcript
- [x] **1b. Frontend integration** — statuses, `/audio`, CORS, m4a/webm uploads
- [ ] **2. Conservative repair** — `app/services/repair.py` is a pass-through
      placeholder; the trained model replaces `PassthroughRepairService`
- [ ] **3. Uncertainty handling** — `app/services/uncertainty.py` is a raw
      probability threshold with no alternatives generator
- [ ] **4. TTS** — `app/services/tts.py` shells out to the OS voice
- [ ] **5. Evaluation harness** — `../evaluation/`

## Repair: constrained hypothesis selection (suggestion-first)

Deployed pipeline:

```
speech → Whisper medium.en → A0 transcript
       → CTranslate2 N-best (beam 12 → top-4 unique alternatives)
       → hybrid list  H1=A0, H2..H5
       → 21-feature extraction per candidate
       → D3-NL constrained selector (tiny MLP, NumPy inference)
       → suggestion-first decision → Revoice UI
```

**The selector can only choose from actual Whisper hypotheses. It does not
freely generate replacement sentences.** A runtime invariant re-verifies that
every surfaced string is a member of the utterance's own hypothesis list.

**Suggestion-first policy (why it exists):** the research evaluation showed
that optimal switching thresholds vary substantially across speakers, and a
single global threshold did not preserve the required safety/precision
properties for unknown speakers. The deployed product therefore NEVER silently
replaces the transcript:

- selector prefers A0 → `KEEP_A0`: existing behaviour, auto-speak.
- selector prefers an alternative → `UNCERTAIN`: `repaired_text` stays A0 and
  the preferred hypothesis is returned as `suggested_text` (also
  `alternatives[0]`). The app shows "I think you said …" with one-tap
  confirmation; only after the user confirms does that text get spoken.
  Automatic switching is disabled (`policy.auto_switch_enabled=false` in the
  artifact). `suggestion_tau` (0.35) grades suggestion strength only.

Future personalization could use accumulated user confirmations to calibrate
automatic repair per speaker; that is explicitly not built.

**Research vs product:** the citable research result is the frozen D2 LOSO
evaluation (WER 0.3175 → 0.2849, preservation 98.9%, 0% unsupported
generation). The deployed weights are the D3-NL development-stage selector
chosen for its stronger safety profile; it is not independently validated on
unseen speakers, and fold-specific research thresholds do not describe this
deployed policy.

Configuration (`.env`): `WHISPER_MODEL=medium.en` (all quoted repair metrics
were measured on medium.en), `REPAIR_BACKEND=selector|passthrough|none`
(`none` = exact pre-selector behaviour; any artifact problem also fails closed
to it), `REPAIR_MODEL_PATH=app/assets/revoice_selector_v1.json`,
`REPAIR_SWITCH_MARGIN` (optional suggestion-strength override; logged loudly).
