# D2 deployment plan — Revoice conservative hypothesis selector

Status: **plan only; nothing integrated.** D2 is frozen as the final hackathon
research system; D3 audit/design preserved as future work.

Backend inspected read-only at `~/Downloads/revoice/speech-repair/backend/`
(it moved from Desktop): FastAPI, `ASRService` ABC (faster-whisper, default
`base.en`), a `RepairService` ABC with NoOp/Passthrough implementations already
waiting for "the real model", `uncertainty.py`, and a response schema that does
NOT yet include the `alternatives`/`decision`/`repair_available` fields from
`Hackaton2026/API_CONTRACT.md`.

## 1. Frozen artifacts required for deployment

| artifact | source | role |
|---|---|---|
| N-best decode recipe | `scripts/nbest_generate.py` (frozen: beam 12, num_hyp 8 → top-5 unique; Silero VAD trim; loop-collapse; dedupe on normalized text) | hypothesis generation |
| Hybrid-list rule | `scripts/flant5_loso.py` / `exp3_d1.py` (A0 prepended, ct2 top-4 unique after) | list construction |
| Feature extractor | `exp3_d1.build_rows` feature block (16 features) | selector input |
| Selector recipe | `exp3_d1.train_fold` (conditional logit, seed, 300 epochs, Adam lr 0.05, wd 1e-3, standardization) | final training |
| Decision policy | `exp3_d2.apply_tau` (margin rule, UNCERTAIN bookkeeping) | conservative decision |
| Per-fold weights (×8) | `d1_evaluation.json` `fold_weights` | sanity reference for the final model |
| Expected-performance numbers | `d2_evaluation.json` | what we honestly quote |
| Labeled dataset | `oracle_rows.csv` + `repair_pairs.csv` + `nbest_cache.jsonl` | final-model training |

What ships to the backend is ~3 small files: a weights JSON (16 floats + 16
means + 16 stds + τ), a feature module (~150 lines, pure Python/NumPy — no
torch at inference), and an N-best module. TORGO data never ships.

## 2. Final deployment selector — no arbitrary fold choice

LOSO validated the *procedure*; the deployed model uses the standard next step:
**retrain once on all 683 labeled rows with the byte-identical frozen recipe**
(same features, seed, epochs, optimizer, standardization fit on the full set).
Rationale: every fold's model was trained the same way on 6 speakers; the final
model is the same estimator on 8. No fold is privileged; nothing new is tuned.

Two sanity gates before accepting the final weights (development-data checks,
not new evaluation): (a) weight signs/magnitudes consistent with the 8 stored
fold weight vectors; (b) resubstitution behaviour sane (switch rate ~9±4%,
preservation ≥99% on the 274 A0-correct rows). Quoted performance remains the
LOSO numbers (WER 0.2849, 47/10, 77% precision) — never the resubstitution fit.

## 3. Exact backend pipeline

```
audio upload
  → existing audio.py preprocessing
  → ASRService.transcribe() (production Whisper — A0 + segment confidence)   [exists]
  → NBestService: reuse the SAME loaded WhisperModel:
      VAD-trim → features → model.model.generate(beam 12, N8, scores)
      → normalize → loop-collapse → dedupe → top-4 unique non-A0             [new]
  → hybrid list [A0, H2..H5]
  → feature extraction (16 features, pure Python)                            [new]
  → selector: softmax(w·x), margin = P(best alt) − P(A0)                     [new]
  → decision:
      margin ≥ τ           → SWITCH  → repaired_text = chosen hypothesis
      argmax≠A0, margin<τ  → UNCERTAIN → repaired_text = A0, flagged
      argmax=A0            → KEEP_A0 → repaired_text = A0
  → response mapping (additive fields per API_CONTRACT.md):
      KEEP_A0/SWITCH → status success, decision "high"|"medium" (see §5)
      UNCERTAIN      → status uncertain, decision "low",
                       alternatives = remaining unique hypotheses (≤3)
      repair_available = true; unsupported generation impossible by construction
  → existing TTS + response                                                   [exists]
```

Fits the existing architecture exactly: one new `SelectorRepairService`
implementing the existing `RepairService` ABC, plus one N-best helper beside
`asr.py`. The frontend already renders `alternatives`/`decision` (shipped
earlier in `Hackaton2026`/`~/Downloads/revoice/frontend`).

## 4. Deployment threshold without a per-speaker validation fold

There is no validation speaker for an unknown user, and D2.1 proved per-speaker
thresholds don't transfer. Deployment policy:

- **One global τ**, chosen on the full labeled set with the frozen priority
  rule (preservation ≥99% → ratio ≥4:1 → min WER → largest τ). This is
  development-data calibration and is labelled as such; expected field
  behaviour is quoted from LOSO, where per-fold τ averaged ~0.05–0.1 for
  speakers with usable switches.
- τ is an env var (`REPAIR_SWITCH_MARGIN`), so field testing can raise it
  toward pure-KEEP_A0 without a redeploy.
- Known limitation, stated: a global τ will be conservative for easy speakers
  (correct behaviour) and slightly suboptimal for hard speakers. Per-user
  adaptation is future work, not hackathon scope.

## 5. Honest confidence exposure

Never exposed: softmax probability as "probability correct" (uncalibrated).
Exposed via existing/contracted fields:

- `confidence`: unchanged — the ASR segment confidence (existing semantics)
- `decision`: "high" = KEEP_A0 with A0-agreeing list (all hypotheses identical)
  or no alternatives; "medium" = SWITCH or KEEP_A0 with disagreeing list;
  "low" = UNCERTAIN. These are rule-based bands, not probabilities.
- `alternatives[].confidence`: **null** — we have no calibrated per-candidate
  score, and the contract explicitly allows null rather than invented numbers.
- Internal logs only: margin, consensus stats, switch/keep decision — for field
  debugging, never rendered to the user.

## 6. Latency, memory, dependencies (measured, not guessed)

| component | measured basis | estimate |
|---|---|---|
| Whisper medium.en A0 (CPU int8) | 6.4 s mean on TORGO clips; shorter for short phrases (~2–4 s for 3–5 s utterances) | dominant cost |
| N-best generate (VAD-trimmed, reused model) | 1.7–1.9 s/clip measured | +~2 s |
| features + selector + decision | measured <1 ms | negligible |
| **added latency vs A0-only** | | **≈ +2 s** |
| memory | medium.en int8 ≈ 3.6 GB peak (measured) | fits the Mac; phone never runs models |
| runtime deps | faster-whisper/ctranslate2 (already installed), NumPy | **no torch, no transformers, no sklearn at inference** |
| model artifact | 16 weights + scaler + τ | < 2 KB JSON |

⚠️ **Config change required:** experiments used `medium.en`; backend default is
`base.en`. Deployment sets `WHISPER_MODEL=medium.en` (env only). All D2 numbers
are meaningless under base.en. First-run model download ≈ 1.5 GB.

## 7. Minimal backend changes (for the backend owner; nothing done yet)

1. `app/services/nbest.py` — new (~120 lines): frozen decode + list build
2. `app/services/selector_repair.py` — new (~150 lines): features + weights +
   decision; implements the existing `RepairService` ABC
3. `app/models/schemas.py` — additive fields: `alternatives`, `decision`,
   `repair_available` (already specified in API_CONTRACT.md; frontend tolerates
   absence and presence)
4. `app/api/routes.py` — call selector after ASR; populate new fields
5. `app/config.py` — `REPAIR_BACKEND=selector|passthrough|none`,
   `REPAIR_SWITCH_MARGIN`, ship with `WHISPER_MODEL=medium.en`
6. weights JSON under `app/assets/`
Nothing else changes; TTS, audio handling, error envelope untouched.

## 8. Physical-iPhone end-to-end test plan

Setup: backend on the Mac (`WHISPER_MODEL=medium.en`, selector on), Expo app on
the physical iPhone, `EXPO_PUBLIC_API_URL=http://<Mac LAN IP>:8000`,
`DEMO_MODE=false`, same Wi-Fi.

| test | speech | expected |
|---|---|---|
| T1 clear sentence ×5 | "could you bring me my glasses" | KEEP_A0, status success, spoken immediately, no regression vs today |
| T2 deliberately mumbled word ×5 | slurred "glasses" | either safe KEEP_A0 or a SWITCH whose text is verbatim one of the logged hypotheses |
| T3 near-silence / tap | — | retry path unchanged |
| T4 long sentence (~15 s) | reading passage | end-to-end < 15 s, no timeout (frontend cap 45 s) |
| T5 alternatives UI | any UNCERTAIN result | prediction shown dominant, alternatives are complete sentences, one-tap speak works |
| T6 hallucination probe ×10 | varied difficult speech | grep backend logs: every `repaired_text` ∈ logged hypothesis list — **must be 10/10** |
| T7 latency log | all above | record A0 time vs total; added cost ≈ +2 s |
| T8 rollback drill | mid-session | flip `REPAIR_BACKEND=none`, restart, verify A0-only behaviour |

Pass = T6 at 100%, T1 with zero harmful changes, T8 clean.

## 9. Rollback path

- `REPAIR_BACKEND=none` (or `passthrough`) → selector and N-best fully bypassed,
  A0-only responses with `repair_available=false`; one env change + restart,
  no code path removed.
- Frontend needs nothing: it already treats missing `alternatives`/`decision`
  as legacy responses.
- Worst-case in-field failure mode is bounded by construction: output is always
  a real Whisper hypothesis, so rollback urgency is about *quality*, never
  about invented speech.
