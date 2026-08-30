# Experiment 3 — conservative hypothesis selector (Stages 1–2, DESIGN ONLY)

Status: **Stage 1 (oracle audit) complete; Stage 2 (design) frozen; nothing
trained.** All prior experiments frozen and untouched. All numbers below come
from cached artifacts only — no audio touched, no new transcription.

## Stage 1 — selector oracle audit (683 dysarthric sentences, hybrid H1–H5)

Hybrid list identical to System C: H1 = production 1-best (A0), H2–H5 = frozen
ct2 top-4 unique. Mean list length 3.24.

| quantity | value |
|---|---|
| A0 WER | 0.3175 |
| **hybrid selection oracle WER** | **0.2132** (gain +0.1043) |
| oracle exact match | 50.5% (A0: 40.1%) |
| reference exactly in list | 50.5% |
| A0 already oracle-best | **68.7%** |
| better candidate exists in H2–H5 | **31.3%** |
| multiple hypotheses tie for best | 18.7% (mean tied 1.35) |
| **oracle vs System C (0.2723)** | **oracle better by 0.0591** |

Per speaker (A0 → oracle):

| speaker | tier | n | A0 | oracle | gain | A0-best |
|---|---|---|---|---|---|---|
| M03 | easy | 95 | 0.0242 | 0.0156 | +0.009 | 96% |
| F04 | easy | 68 | 0.0477 | 0.0239 | +0.024 | 88% |
| F03 | medium | 139 | 0.1326 | 0.0901 | +0.043 | 81% |
| M05 | medium | 115 | 0.2752 | 0.1830 | +0.092 | 68% |
| M02 | hard | 92 | 0.4366 | 0.2763 | +0.160 | 54% |
| F01 | hard | 20 | 0.5761 | 0.4674 | +0.109 | 55% |
| M01 | hard | 89 | 0.6335 | 0.4239 | +0.210 | 46% |
| M04 | hard | 65 | 0.7633 | 0.5163 | +0.247 | 38% |

Key subsets:

- **Unseen prompts (55):** 0.1466 → 0.1114 oracle (**+0.035 gain**) — selection
  has positive headroom exactly where C regressed, and cannot hallucinate.
- **LOW (119):** 1.0433 → 0.6502 oracle (**+0.393**) with the reference in the
  list only 8% of the time — huge safe headroom from picking a *less wrong*
  hypothesis.
- **A0-already-correct (274):** the oracle keeps A0 **100%** of the time —
  under selection, preservation failures can only be *active wrong switches*,
  never drift.

**Label distribution** (rule: min WER; H1 ties → KEEP_A0; else best decoder
rank, i.e. lowest index — the spec's edit-distance tie-break never fires
because rank order is total):

| label | count | share |
|---|---|---|
| KEEP_A0 | 469 | 68.7% |
| H2 | 125 | 18.3% |
| H3 | 38 | 5.6% |
| H4 | 29 | 4.2% |
| H5 | 22 | 3.2% |

Full breakdowns in `oracle_analysis.json` / `label_distribution.json` /
`oracle_rows.csv`.

**Headroom verdict (question 15): YES — training is justified.** The selection
ceiling (0.2132) beats C by 0.059, the KEEP_A0-heavy label prior matches the
product's conservative bias, and the two subsets where C failed (unseen
prompts, LOW) are precisely where selection is safest.

**Honesty note on the stretch goal:** aggregate WER ≤ 0.15 is **unreachable
even by a perfect selector** (oracle 0.2132). It stays recorded as non-binding
but cannot be met by D1/D2 on this list; only a D3-style local repair or a
better hypothesis list could approach it.

## Stage 2 — frozen design

### Architecture (D1 core): per-candidate feature ranker, regularized logistic regression

Not Flan-T5. Selection is a ranking/classification problem over ≤5 grounded
candidates with ~500 labeled rows per fold; the simplest model that can use
ASR evidence and produce probabilities wins:

| option | params | verdict |
|---|---|---|
| Flan-T5-base constrained to an index token | 250 M | rejected: 990 MB, ~1 s latency, massive capacity for a 5-way choice, poorly calibrated token scores, highest overfit risk |
| small encoder + classification head (e.g. MiniLM) | 22 M | rejected for round 1: still fine-tunes millions of params on ~500 rows; no natural ASR-score fusion |
| pairwise ranking net | ~10⁵ | viable but more machinery than needed |
| **engineered features + L2 logistic regression (scored per candidate, softmax over the list)** | **~40** | **chosen**: seconds to train, trivially calibratable, interpretable, ~µs inference, tiny artifact, runs anywhere behind FastAPI |

Scoring (clarified 2026-08-30): a **conditional-logit / listwise choice
model** — `score(Hi) = w · x_i` with the SAME weight vector `w` applied to
every candidate, `P(Hi) = softmax over the utterance's list`. No
position-specific parameter sets: rank is an input *feature*, never a separate
class head, so the model learns "what makes a hypothesis supported", not
"which numbered slot tends to win". sklearn's LogisticRegression cannot
express this shared-scorer form, so the model is implemented directly as a
tiny deterministic PyTorch module (a single linear layer, no bias —
list-constant features cancel in the softmax and are excluded). Exact trainable
parameter count is reported by the training script. Label = the Stage-1 rule.
No new dependencies needed.

### Inference-time features (audited: every one available in the real backend)

Per candidate *i* in its list:
1. `is_a0` (H1 flag)
2. list rank (0 for A0; ct2 rank for H2–H5)
3. **Two strictly separate score families (corrected 2026-08-30 — production
   confidence and ct2 log-scores are different quantities and are NEVER
   normalized against or compared with each other):**
   - *A0-specific:* production engine confidence; confidence-missing flag.
     Nonzero only on the A0 candidate.
   - *ct2-specific:* raw decoder score; score relative to the best ct2
     candidate; gap to the neighbouring ct2 candidate; per-word-normalized
     score — all computed WITHIN the ct2 candidates only, zero on A0, with a
     ct2-score-missing indicator.
   Cross-hypothesis evidence uses only genuinely comparable quantities:
   lexical consensus, word overlap, edit distance, length, rank,
   disputed-word agreement.
4. word-level edit distance to A0, absolute and length-normalized
5. candidate length in words; ratio to the list's median length
6. **consensus support**: mean word-overlap F1 against the other hypotheses;
   fraction of the candidate's words appearing in ≥2 other hypotheses
7. exact-duplicate count (how many raw ct2 hypotheses normalize to this text)
8. **disputed-word support**: of the words where the candidate differs from A0,
   the fraction attested in at least one other hypothesis
List-level (shared): number of unique hypotheses, mean pairwise WER of the
list, whether all hypotheses agree.

Banned and absent: ground truth, reference WER, CORRECT/HIGH/MEDIUM/LOW labels,
speaker identity, prompt identity. Labels appear only in the training loss.

### KEEP_A0 bias (D2) — switching margin

D1 = plain argmax (no threshold) — reported as the unbiased selector.
D2 = switch to the top non-A0 candidate only if
`P(candidate) − P(A0) ≥ τ`; otherwise output A0.
τ is chosen **per fold on that fold's validation speaker only**, by grid search
τ ∈ {0.00, 0.05, …, 0.90}: **MINIMIZE validation WER subject to validation
correct-input preservation ≥ 99%** (corrected 2026-08-30 — the earlier text
said "maximizing", which was wrong). Deterministic conservative tie-break:
among thresholds whose validation WER is effectively identical (within 1e-9),
**the LARGEST τ wins** — more KEEP_A0 behaviour. Test speakers never touch
threshold choice.

### UNCERTAIN rule (D2), fixed now

`UNCERTAIN` ⇔ the selector's argmax is not A0 **but** the margin is below τ
(evidence existed, abstained). Textual output for scoring is **always A0** in
that case — abstention is bookkept separately and can never touch ground truth.
Risk-vs-coverage: sweep τ on validation predictions → `risk_coverage.csv`.
Raw softmax probabilities are NOT presented as calibrated confidence; a
validation-only reliability check is reported, and with n as small as 20 (F01)
calibration numbers will be labelled indicative only.

### LOSO

Identical folds, identical validation-neighbour rule as B/C. All selection,
thresholds, calibration on train+validation of each fold. Control set scored
per fold. Hard constraint verified mechanically: output ∈ {H1…H5} for every
row; **unsupported-generation rate must equal exactly 0% or it is a bug**.

### Frozen success criteria (Stage 12; unchanged after this point)

1. D2 aggregate WER ≤ **0.2723** (C) — with the tradeoff clause as written
2. D2 aggregate WER clearly < **0.3175** (A0)
3. unsupported-generation rate = **0%** (hard)
4. correct-input preservation ≥ **99%**
5. control preservation ≥ **99%**
6. improved:worsened ≥ **4:1**
7. edit precision ≥ **75%**
8. unseen-prompt WER not worse than A0
9. unseen-prompt improved ≥ worsened
10. LOW WER not worse than A0
11. harmful-LOW-switch threshold: pre-registered in Stage 4 from
    train/validation behaviour only, before any test evaluation
12. ≤1 of 8 speakers meaningfully degraded (tolerance +0.005 WER)
Stretch (non-binding, already known unreachable by selection alone): ≤ 0.15.

### Cost estimates

| stage | estimate |
|---|---|
| feature build (683 + 194 rows, cached text) | < 1 min |
| D1 training, 8 folds | **< 1 min total** |
| D2 threshold grid on validation folds | < 1 min |
| full evaluation + report | ~2 min |
| selector inference latency | **< 1 ms/utterance** (dominant cost stays Whisper+N-best: ~6 s + ~1.7 s) |
| model artifact | a few KB per fold |

Deployment note: N-best generation can later be optimized (smaller beam,
faster compute type, streaming) without changing this experiment — the
selector consumes hypothesis text + scores regardless of how they were decoded.

### D3 (not started)

Decision deferred until D1/D2 results. The audit already bounds it: 49.5% of
rows lack the exact reference in the list, and LOW oracle still leaves 0.65
WER — so meaningful headroom beyond selection exists, but D3 is only worth
proposing if D1/D2 demonstrate the safety machinery holds. Requires separate
approval.

**STOPPED. Awaiting approval to run Stage 3 (train/evaluate D1).**
