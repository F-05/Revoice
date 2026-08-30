# Experiment 2 (redesigned): N-best → Flan-T5 generative repair — DESIGN ONLY

Status: **proposal, nothing trained**. The experiment-1 checkpoint is not
integrated. Frontend/backend untouched. Experiment-1 outputs preserved; all
experiment-2 artifacts go to new paths (`data/large_torgo/nbest/`,
`results/t5_nbest/`, `models/revoice-flant5-nbest/`).

**Revision 2** — incorporates Nathan's corrections of 2026-08-29: an
N-best-decoder-only baseline (A1), the corrected experimental claim (N-best +
Flan-T5 is a *combined* change), frozen N-best parameters, a leakage-safe gate,
and the amended LOW criterion.

**Baseline B is FROZEN as a failed baseline** (per Nathan, 2026-08-29), outputs
preserved unmodified in `results/t5_loso/`: aggregate WER 0.3175 → 0.3225,
12 improved / 638 unchanged / 33 worsened, edit rate 8.6%, edit precision 20%,
correct preservation 94.9%, control degradation +0.0096, control preservation
94.7%, LOW rewritten 24/119. Per speaker: M03 0.024→0.041, F04 0.048→0.052,
F03 0.133→0.144, M05 0.275→0.275, M02 0.437→0.444, F01 0.576→0.592,
M01 0.634→0.641, M04 0.763→0.737 (the sole improvement). Not retrained, not
overwritten.

---

## 1. Prior-work analysis

Everything below was read from the repositories/papers on 2026-08-29. UNKNOWN
means the source does not state it — nothing is inferred.

### GER4Dys (github.com/morenolaquatra/ger4dys)

| aspect | finding |
|---|---|
| ASR | "Whisper-based models" |
| Hypotheses | N-best; N UNKNOWN; generation method UNKNOWN |
| Diversity | "diversity-based hypothesis selection" claimed; mechanism UNKNOWN |
| Repair model | fine-tuned Flan-T5 **with LoRA** |
| Prompt format | UNKNOWN |
| Data | Speech Accessibility Project Challenge (140+ h dysarthric) + TORGO and VoxPopuli as augmentation |
| Correct/catastrophic handling, metrics, confidence | all UNKNOWN |
| **Code available** | **NONE — repo contains only README + a pipeline PNG; "under construction"** |

Usable from GER4Dys: the *architecture direction* only (dysarthric N-best →
Flan-T5+LoRA GER). No code to reuse; nothing about its training recipe is
verifiable today.

### CHSER / GenSEC (github.com/balaji1312/CHSER)

| aspect | finding |
|---|---|
| ASR | Whisper-**base.en**, "zero-shot beam search setting" |
| N-best | not clearly specified in README (UNKNOWN) |
| Repair models | T5-based and Llama2-based correction, **adapter (PEFT) weights** published |
| Prompt/objective/balancing | UNKNOWN from README |
| Data | child speech (MyST etc.), ASR hypotheses paired with verified references |
| Confidence/abstention | UNKNOWN |
| Code | `code/{analysis,dataset_gen,gensec}`, model dirs (3gram, llama2, t5, transformer) |

Usable from CHSER: confirmation that the hypotheses-paired-with-reference
dataset construction we already use is the standard GenSEC recipe; its domain
(child speech) parallels ours (dysarthric) as "atypical-speech GER".

### FlanEC (github.com/MorenoLaQuatra/FlanEC) — the main template

| aspect | finding |
|---|---|
| Data | HyPoradise: 334k+ (N-best list, reference) pairs |
| N-best origin (HyPoradise paper) | large beam search (beam 50–60), de-duplicate, keep **top-5 unique** |
| N | **5** |
| Model | Flan-T5 base / large / XL |
| LoRA | both offered: `train_flanec.py` and `train_flanec_lora.py` |
| Prompt (from `data_classes/hyporadise_dataset.py`) | prefix `"The following is a n-best list of ASR hypotheses for the given audio file:"`, hypotheses as `"{i+1}. {sentence}\n"`, suffix `"The correct transcription is:"`; optional per-hypothesis acoustic scores `"{i+1}. {sentence} ({am_score})\n"` |
| Objective | standard seq2seq cross-entropy to the reference (+eos) |
| Correct-ASR / catastrophic handling | UNKNOWN (no filtering visible in the dataset class) |
| Metrics | WER/CER via HF `evaluate` + Hypo2Trans script |
| Confidence/abstention | UNKNOWN / none visible |

Sources: [HyPoradise paper](https://arxiv.org/pdf/2309.15701), [FlanEC repo](https://github.com/MorenoLaQuatra/FlanEC), [GER4Dys repo](https://github.com/morenolaquatra/ger4dys), [CHSER repo](https://github.com/balaji1312/CHSER).

### What is prior work vs ours vs new

- **From prior work (adapted, credited, not claimed as novel):** N-best GER as
  a task; the FlanEC/HyPoradise prompt serialization; beam-then-dedupe-top-5
  N-best construction; Flan-T5 as the corrector.
- **Already ours (exp 1):** TORGO metadata + verified speaker parsing, headMic
  selection, cached medium.en 1-best, repairability triage, LOSO fold plan,
  control set, unseen-prompt bookkeeping, conservative-repair metrics.
- **Specific to Revoice (new here):** the pre-registered oracle-WER gate before
  any training; conservative-repair success criteria (preservation, edit
  precision, LOW-rewrite accounting); the decision-layer signal design in §9;
  applying the recipe at hackathon scale (683 sentences, 8 speakers) with LOSO.

---

## 2. Existing-project audit (what is reused unchanged)

| asset | state | reuse |
|---|---|---|
| `data/large_torgo/metadata.csv` — 16,552 rows, speakers parsed from filenames, 0 disagreements vs dataset columns | done | unchanged |
| headMic filtering + 683 dysarthric sentences | done | unchanged |
| cached medium.en 1-best (`asr_cache.jsonl`, 877 clips) | done | unchanged — Baseline A and all repairability labels come from it |
| CORRECT/HIGH/MEDIUM/LOW triage (`repair_pairs.csv`) | done | unchanged; labels stay defined on the **1-best**, so categories are comparable across experiments 1→2 |
| control set (194 control-speaker sentences) | done | unchanged |
| unseen-prompt detection (55 rows aggregate under LOSO) | done | unchanged |
| LOSO folds (`loso_folds.csv`, difficulty-neighbour validation rule) | done | unchanged |
| exp-1 T5 pipeline + outputs, exp-2 1-best LOSO (Baseline B) | done / finishing | frozen, never overwritten |

**New artifacts needed:** N-best lists per clip (one new generation pass over
the same 877 clips), Flan-T5 training/eval scripts, new results dirs.

---

## 3. Proposed architecture

```
dysarthric audio (headMic)
  → VAD trim (same Silero VAD faster-whisper uses; matches production's vad_filter=True)
  → Whisper medium.en encoder (ctranslate2), single 30 s window (max clip = 23.2 s)
  → beam search: beam_size=12, num_hypotheses=8, return_scores=True
  → normalize → collapse repetition loops → de-duplicate → keep top-5 unique
  → FlanEC-style prompt
  → Flan-T5 seq2seq repair
  → (future, not this experiment) conservative decision layer → HIGH/MEDIUM/LOW
```

## 4. N-best generation strategy — verified, not assumed

`faster_whisper.transcribe()` cannot return N-best. But the underlying
`ctranslate2.models.Whisper.generate()` (ct2 4.8.1, installed) accepts
`beam_size`, `num_hypotheses`, `return_scores` — true beam-search N-best,
confirmed by reading the installed signature, not docs.

**12-clip probe (train/validation speakers only; test speakers untouched):**
beam=10, N=5 on medium.en int8:

- clips with >1 unique hypothesis: **12/12**
- mean unique hypotheses: **4.42**
- oracle beat 1-best on **10/12** clips
- hypotheses are *linguistically* diverse, exactly the evidence GER needs
  (e.g. M04: "i will continue all the time" / "i would continue all the time" /
  "all the time"; M02: "there's a lot of schoolhouse students and me" /
  "i love school ...")

**Known artifact found by the probe:** the raw ct2 path skips faster-whisper's
temperature-fallback/VAD post-processing, so padded silence produces repetition
loops ("the dolphin swam around our boat" ×9). Mitigations, in order: (a) VAD
trim before feature extraction (production parity), (b) `no_repeat_ngram_size`
left at 0 but a **loop-collapse post-processor** (collapse a whole-phrase
repetition tail into one occurrence) before dedup, (c) `repetition_penalty`
held at 1.0 so we change nothing linguistic. The probe's inflated raw WERs are
this artifact; they do not reflect the production 1-best (which stays the
cached transcript from `transcribe()`).

**Baseline A0 is always the cached production 1-best.** Hypothesis #1 of the
raw N-best pass is reported separately as **A1** (see §9), so decoder effects
are measured, not hidden.

### Frozen N-best configuration

Frozen NOW, from the development probe only, before any LOSO test evaluation.
None of these may be tuned after seeing test-speaker results:

| parameter | frozen value |
|---|---|
| beam_size | 12 |
| num_hypotheses requested | 8 |
| kept | top-5 unique after normalize + collapse + dedupe |
| dedupe key | `normalize_text` output (case/punct-insensitive) |
| VAD trim | Silero VAD, faster-whisper defaults — identical settings to production `vad_filter=True` |
| loop-collapse | reference-independent: within a hypothesis, a phrase of ≥2 words immediately repeated ≥2 times collapses to one occurrence; applied before dedupe; uses ONLY the hypothesis text |
| beam scores | stored in the cache for later decision-layer analysis; **NOT included in the prompt** |
| prompt | FlanEC format verbatim (§7), numbered list, no scores |
| max input / target length | 256 / 128 tokens |

## 5. Oracle-WER gate — leakage-safe version

Under LOSO every speaker is eventually a test speaker, so a gate computed on
all speakers would let test references influence the system. Corrected design:

- **Gate/development set = the 6 speakers of the original exp-1 train+validation
  split** (F03, M01, M02, M03, F01, M04; 500 headMic sentences). F04 and M05
  audio and references are NOT touched for any architecture decision.
- Gate metrics on that set only (final list per Nathan, 2026-08-29):
  unique hypotheses per utterance; % with >1 unique hypothesis (target ≥ 60%);
  average pairwise inter-hypothesis edit distance (WER); cached 1-best (A0)
  WER; A1 (new decoder top-1) WER; **N-best oracle WER**; oracle exact-match
  rate; % utterances whose reference appears EXACTLY in the N-best list;
  oracle gain broken down by speaker and difficulty tier.
- **GATE: proceed only if dev-set oracle WER ≤ dev-set 1-best WER − 0.03
  absolute.** Otherwise STOP and report before any training.
- N-best parameters are already frozen (§4); the gate is a go/no-go decision,
  not a tuning loop. If the gate fails there is no parameter search — the
  result is reported and work stops.
- **Residual leakage, stated plainly:** the frozen parameters came from a
  12-clip probe on dev speakers, and the gate uses dev-speaker references.
  Those 6 speakers still serve as LOSO test speakers in 6/8 folds. The
  parameters are global and coarse (beam width, N, dedupe) rather than
  per-speaker, so the risk is limited — and folds 1 (F04) and 3 (M05) are
  designated **clean folds**: no architecture decision ever saw their data,
  so they carry the most trustworthy numbers.
- Held-out-speaker oracle WER (F04, M05) is computed and REPORTED only after
  the experiment completes; it influences nothing.

## 5b. GATE RESULT (2026-08-29) — PASS, with a structural finding

500 dev clips, frozen config. Aggregate: A0 0.3608, A1 0.4602, ct2 N-best
oracle 0.2991 → **oracle gain +0.0617 > 0.03 threshold: GATE PASSES.**
Diversity: 3.20 unique hyps/utterance, 81.8% multi-hypothesis, mean pairwise
WER 0.318, reference exactly in list 42.8%.

**Structural finding the aggregate hides:** the ct2 decode path is WORSE than
production on easy/medium speakers (M03 A1 0.265 vs A0 0.024; F03 0.421 vs
0.133) and BETTER on hard speakers (M01 0.571 vs 0.634, M04 0.639 vs 0.763).
Oracle gain is entirely a hard-speaker effect (+0.137…+0.218); on easy/medium
speakers even the best ct2 hypothesis is worse than production 1-best
(gain −0.047, −0.065).

**Proposed amendment (needs approval — changes the frozen config):** prepend
the cached production 1-best (A0) as hypothesis #1 of the prompt list, ct2
top-4 unique after it. Dev-set hybrid oracle = **0.2413** (gain +0.1195 vs A0,
double the ct2-only gain), positive for every speaker including M03 (+0.009)
and F03 (+0.043). 88% of rows already contain a hypothesis at least as good as
A0. This also gives the repair model an anchor for the copy/preserve action and
removes the easy-speaker regression risk. A0 is dev-visible data (cached
predictions), so this amendment uses no held-out information.

## 6. Model choice

| option | params | est. train/fold (MPS) | note |
|---|---|---|---|
| flan-t5-small | 80 M | ~3 min | cheapest; same capacity class that just failed twice on 1-best, but the input change is the hypothesis being tested |
| **flan-t5-base** | 250 M | ~8–12 min | FlanEC's smallest published setting; instruction-tuned, so the prompt format is in-distribution; still trains 8 folds in ≤ 1.5 h |
| flan-t5-large | 780 M | ~30 min+ | rejected: overfitting risk on ≤500 rows, slow, no evidence it is needed |

**Recommendation: flan-t5-base, full fine-tuning, no LoRA.** LoRA's value is
memory/parameter efficiency at scale; at 250 M params on a 16 GB Mac it adds a
hyperparameter surface without need. (FlanEC publishes both paths; its
LoRA-vs-full comparison details are UNKNOWN from the repo, so no claim is made
about which is better — we choose full FT for simplicity at this size.)
flan-t5-small is the pre-declared fallback if base overfits (validation WER
rising while train loss falls).

## 7. Training-data construction

FlanEC prompt, verbatim structure:

```
The following is a n-best list of ASR hypotheses for the given audio file:
1. <hyp 1>
2. <hyp 2>
...
The correct transcription is:
```

Target: normalized ground truth (+ eos). If only one unique hypothesis exists,
the list has one entry — the model must still learn "nothing to choose = keep".

Balancing (unchanged from the approved 1-best rerun). **Corrected claim:**
experiment 2 changes BOTH the input representation (1-best → N-best) AND the
repair model (t5-small → flan-t5-base). It therefore tests the **combined
N-best + Flan-T5 architecture**, not the input representation in isolation.
**Optional ablation B′ (described, NOT run without approval):** train
flan-t5-base on 1-best input with the same recipe (~1.5–2 h). C vs B′ would
then isolate the input representation under a fixed model, and B′ vs B the
model change. Composition:

- keep all CORRECT rows (preservation must be taught)
- HIGH/MEDIUM ×2 oversampling
- **LOW excluded from training** — prior work gives no guidance here (all three
  repos: UNKNOWN), so we keep our own rule: a text model must not be taught to
  reconstruct sentences from evidence-free input
- LOW retained in validation/test for abstention analysis
- repairability labels stay computed on the cached 1-best for cross-experiment
  comparability; a per-fold note will report how many LOW rows contain the
  truth somewhere in their N-best (the "N-best rescues" count)

Balancing decisions use train/validation folds only.

## 8. LOSO design — unchanged

8 folds, every speaker tested exactly once, validation = difficulty-rank
neighbour (pre-registered in `experiment2_design.json`), control set evaluated
per fold, no test-fold peeking for any tuning decision. Difficulty tiers
easy (M03, F04) / medium (F03, M05) / hard (M02, F01, M01, M04) are reported
**in addition to**, never instead of, per-speaker numbers.

## 9. Metrics + conservative-repair evaluation

Everything from the approved rerun: aggregate + per-speaker WER, exact match,
improved/unchanged/worsened, improved:worsened ratio, edit rate, edit
precision, correct-input preservation, control WER + preservation, LOW
behaviour, unseen-prompt subset (55 rows), per-tier summaries.

Comparisons reported side by side:
- **A0**: cached production medium.en 1-best (faster-whisper `transcribe()`)
- **A1**: the ct2 N-best pipeline's **top-ranked hypothesis, no repair model**.
  The raw ct2 decode path differs from production (no temperature fallback,
  different silence handling), so A1 separates "the decoder changed" from
  "the repair helped". Costs nothing extra: it is hypothesis #1 of the same
  N-best cache.
- **B**: production 1-best → exp-1-style T5 under LOSO (complete, frozen)
- **C**: N-best → Flan-T5 generative repair (proposed)

Required comparisons: **A0 vs A1** (decoder effect), **A1 vs C** (repair effect
on the same decode), **A0 vs C** (end-to-end effect), **A0 vs B** (old
architecture, already measured).

## 10. Pre-registered success criteria (same thresholds, one review note)

1. Aggregate repaired WER ≤ 1-best − **0.010** absolute
2. No meaningful degradation (ΔWER ≤ +0.005) on ≥ **6/8** speakers AND clear
   improvement (ΔWER ≤ −0.010) on ≥ **2** speakers having ≥10 HIGH/MEDIUM rows
3. improved:worsened ≥ **3:1**
4. ≥ **5%** of HIGH/MEDIUM (reparable) inputs improved
5. correct-input preservation ≥ **98%**
6. control WER degradation ≤ **0.002**
7. control preservation ≥ **99%**
8. unseen-prompt WER not worse than 1-best on those 55 rows
9. **LOW behaviour (amended per Nathan, 2026-08-29):**
   - LOW rewrite rate ≤ **10%**
   - **no aggregate WER degradation on the LOW subset** (LOW WER after ≤ before)
   - every LOW rewrite is itemised in the report — sample id, before/after
     text, and whether it improved, worsened, or preserved row WER
   - LOW remains excluded from repair training

All other thresholds kept exactly as pre-registered. Nothing changes after
results.

## 11. Estimated runtime / resources (measured bases, not guesses)

| stage | basis | estimate |
|---|---|---|
| N-best generation, gate set (568 clips) | probe ≈ 4–6 s/clip incl. VAD | **~50 min** |
| Gate analysis | pandas | minutes |
| N-best, test+control (503 clips, only if gate passes) | same | **~45 min** |
| Flan-T5-base training | t5-small measured 2.3 min/fold; ×4 params, longer inputs | **~8–12 min/fold, ~1.5 h for 8 folds** |
| Evaluation per fold (≈380 beam-4 generations) | measured 11 s for 377 on t5-small; ×~4 | **~1 min/fold** |
| Inference latency (future demo) | medium.en ≈ 6 s/clip CPU + N-best overhead ≈ ×1.3 + flan-t5-base ≈ 1 s | **~9 s/utterance** — usable for a demo, slow for production; noted as risk |
| Memory | medium.en int8 ≈ 3.6 GB peak + flan-t5-base fp32 ≈ 1 GB | fits 16 GB |
| Checkpoint | flan-t5-base ≈ **990 MB** fp32 (t5-small was 242 MB) | 8 folds kept as results only; final model retrained once on a chosen split if integration is ever approved |

## 12. Implementation plan (exact, in order)

1. `scripts/nbest_generate.py` — VAD trim → ct2 `generate(beam_size=12,
   num_hypotheses=8, return_scores=True)` → normalize → loop-collapse → dedupe
   → top-5 unique + scores → `data/large_torgo/nbest/nbest_cache.jsonl`
   (incremental, resumable). Gate set first.
2. `scripts/nbest_gate.py` — diversity + oracle metrics on gate set →
   `results/t5_nbest/nbest_gate.json`. **STOP here and report if gate fails.**
3. `scripts/flant5_loso.py` — same fold loop as `t5_loso.py`, FlanEC prompt,
   flan-t5-base, 5 epochs, ×2 oversampling, checkpoint on validation WER.
4. `scripts/flant5_loso_report.py` — criteria scorer (shared logic with the
   1-best scorer, plus LOW-rewrite ≤15% rule and A/B/C comparison table).
5. Final report with A vs B vs C and an integration recommendation.

## 13. Risks

- **Tiny training set** (~500 effective rows/fold) under a 250 M model:
  overfitting watched via per-epoch validation WER; flan-t5-small fallback
  pre-declared.
- **Prompt memorisation** unchanged from exp 1 (156 unique prompts): mitigated
  only by reporting unseen-prompt rows separately; not solvable at this size.
- **Repetition-loop artifact** in raw N-best: mitigated by VAD trim +
  loop-collapse; residual risk that some hypothesis slots are wasted.
- **Oracle gate may fail** — that is the gate working, not a failure of the
  experiment; ~50 min of compute buys the answer.
- **Latency** (~9 s/utterance) is acceptable for a demo, not for production.
- **Copying remains a strong local optimum**; N-best changes the evidence, not
  the objective. If C also copies, the honest conclusion is that this data
  volume cannot train a repair model, and the decision layer should instead
  consume raw N-best agreement signals (no trained repair).

## 14. Eventual integration design (design only — NOT built now)

Pipeline: audio → medium.en N-best → repair → decision layer → frontend
HIGH/MEDIUM/LOW (contract already shipped in `Hackaton2026/API_CONTRACT.md`).

Generation probability is not calibrated confidence and will not be presented
as such. Candidate decision signals, to be validated on validation folds only:

- **N-best agreement**: all 5 hypotheses identical after normalization → strong
  HIGH signal; probe shows agreement correlates with easy speakers
- **beam score gap** between hypothesis 1 and 2 (ct2 `return_scores`)
- **repair-in-list**: repaired sentence ∈ N-best list → evidence-backed edit;
  repair ∉ list → treat as guess (cap at MEDIUM)
- **edit magnitude**: word-level distance 1-best → repair; large edits demote
- **ASR segment confidence** (already exposed by the backend today)
- if data ever allows: a tiny calibrated classifier on validation folds mapping
  these signals → HIGH/MEDIUM/LOW; until then, thresholded rules

`alternatives[]` in the API maps naturally to the deduped N-best/top candidates,
which is exactly what the frontend already renders.

---

## RECOMMENDED EXPERIMENT 2

```
ASR:               faster-whisper medium.en int8 (cached 1-best = Baseline A; unchanged)
N-best:            ct2 beam_size=12, num_hypotheses=8 → VAD trim, loop-collapse,
                   dedupe → top-5 unique + beam scores
Repair model:      google/flan-t5-base, full fine-tune (no LoRA);
                   flan-t5-small pre-declared fallback
                   NOTE: tests the COMBINED N-best + Flan-T5 change vs B;
                   optional ablation B′ (flan-t5-base on 1-best) needs approval
Fine-tuning:       5 epochs, lr 5e-5, batch 8, max_len 256 input / 128 target,
                   FlanEC prompt, checkpoint on validation-fold WER
Dataset:           683 dysarthric headMic sentences + 194 control (all cached)
Mic:               headMic only (arrayMic reserved for a later augmentation exp)
Training examples: CORRECT + HIGH×2 + MEDIUM×2 per fold (labels from 1-best)
LOW handling:      excluded from training; kept in eval; rewrite rate ≤15% criterion
Evaluation:        8-fold LOSO, criteria §10; A0 vs A1 vs B vs C
                   (A1 = frozen N-best pipeline top-1, no repair;
                    folds F04 and M05 flagged as fully-clean folds)
Success gate:      BEFORE training — dev-set (6 exp-1 train/val speakers)
                   oracle WER ≤ dev-set 1-best WER − 0.03 absolute, params
                   frozen first, else STOP and report
Estimated runtime: ~50 min gate generation → gate report → (if pass)
                   ~45 min remaining N-best + ~1.5–2 h LOSO training/eval
                   ≈ 3–3.5 h total
```
