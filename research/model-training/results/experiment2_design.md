# Experiment 2 — proposed design (NOT YET RUN)

Awaiting approval. No model has been trained under this design, and the
experiment-1 checkpoint is not integrated anywhere.

One factor changes versus experiment 1: **the evaluation design**. ASR stays
`medium.en`, the repair model stays `t5-small` at 5 epochs, arrayMic stays out.

## 1. Speaker difficulty (already measured, not re-derived)

From the cached `medium.en` predictions. This is ASR difficulty, not a clinical
severity rating — TORGO's severity labels are absent from this dataset.

| speaker | sex | n | ASR WER | exact | severe |
|---|---|---|---|---|---|
| M03 | male | 95 | 0.024 | 85.3% | 0.0% |
| F04 | female | 68 | 0.048 | 79.4% | 0.0% |
| F03 | female | 139 | 0.133 | 52.5% | 5.8% |
| M05 | male | 115 | 0.275 | 34.8% | 13.9% |
| M02 | male | 92 | 0.437 | 19.6% | 22.8% |
| F01 | female | 20 | 0.576 | 0.0% | 25.0% |
| M01 | male | 89 | 0.634 | 7.9% | 41.6% |
| M04 | male | 65 | 0.763 | 1.5% | 49.2% |

Difficulty spans a factor of ~32 in WER. Any single split of 8 speakers is
dominated by which speakers land where.

## 2. Proposed design: leave-one-speaker-out cross-validation

8 folds. Fold *k* tests the *k*-th speaker in ascending ASR-difficulty order.

**Validation rule (fixed, positional, decided before fitting anything):**
validation is the difficulty-rank *neighbour* of the test speaker — the next
harder speaker, or for the hardest speaker the next easier one. This guarantees
validation resembles test in difficulty, which is precisely what failed in
experiment 1 (validation held the two hardest speakers, test held the easiest).

| fold | test | sex | test WER | n | validation | val WER | train n | eligible | unseen-prompt n |
|---|---|---|---|---|---|---|---|---|---|
| 0 | M03 | m | 0.024 | 95 | F04 | 0.048 | 520 | 401 | 1 |
| 1 | F04 | f | 0.048 | 68 | F03 | 0.133 | 476 | 365 | 2 |
| 2 | F03 | f | 0.133 | 139 | M05 | 0.275 | 429 | 334 | 42 |
| 3 | M05 | m | 0.275 | 115 | M02 | 0.437 | 476 | 394 | 5 |
| 4 | M02 | m | 0.437 | 92 | F01 | 0.576 | 571 | 478 | 2 |
| 5 | F01 | f | 0.576 | 20 | M01 | 0.634 | 574 | 497 | 1 |
| 6 | M01 | m | 0.634 | 89 | M04 | 0.763 | 529 | 479 | 1 |
| 7 | M04 | m | 0.763 | 65 | M01 | 0.634 | 529 | 479 | 1 |

Aggregate: **683 test rows** (every dysarthric sentence tested exactly once, by
a model that never saw that speaker) and **55 unseen-prompt rows**, up from 11.

## 3. Variance, stated plainly

With 8 speakers, any single split is high-variance. Per-fold results will differ
a lot, because the folds genuinely differ in difficulty — fold 5 tests 20
sentences from a speaker with 0% ASR exact match. Per-fold numbers are therefore
reported as a distribution, never as one headline. The aggregate is a
speaker-weighted mean, and the per-fold spread is reported alongside it.

## 4. Why cross-validation over one split — and what it costs

**For:** with LOSO there is no test set to choose, so the selection bias the
brief warns about cannot occur. Every speaker is tested exactly once, all 683
sentences contribute, the male/female imbalance evens out across folds instead
of having to be balanced inside one small split, and a per-speaker result curve
falls out — which is the practically useful thing to know (does repair help mild
speakers, severe speakers, or nobody?).

**Against:** 8 training runs instead of 1; folds are not independent (they share
training speakers), so the spread across folds understates true variance and
must not be reported as a confidence interval; and the aggregate mixes speakers
of very different difficulty, so it must always be shown with the per-speaker
breakdown.

**Cost:** experiment 1 trained in 2 min 18 s and evaluated in 11 s. Eight folds
is roughly **20–25 minutes**, with **no ASR re-transcription at all**. The
scientific gain is large and the cost is small, so LOSO is the recommendation.

## 5. Training composition

A correction to my earlier explanation: experiment 1 was **not** dominated by
correct examples — it held 179 correct against 170 erroneous, already about
1:1. Rebalancing alone therefore does not explain the copying, and would not on
its own fix it. Copying is a strong local optimum: it is exactly right for half
the data and still closer than a wrong guess on much of the rest.

Proposal, keeping all three categories:

- keep every CORRECT row (preservation must still be taught),
- **oversample HIGH/MEDIUM rows ×2 per epoch**, moving the correct share from
  0.35–0.56 down to **0.21–0.38** depending on fold,
- keep LOW out of training (per instruction), retained in evaluation,
- report an **edit rate** and **edit precision** so copying is visible as a
  number rather than inferred: experiment 1 scored edit rate 4.9% (9/183) and
  edit precision 17% (1 of 6 edits helped).

Oversampling changes only how often an example is shown, never its content, and
is applied inside the training loop after the fold split, so it cannot leak.

## 6. Pre-registered success criteria

Declared before training. If these are not met, the answer is "do not
integrate", regardless of how any individual fold looks.

1. **Beats ASR alone.** Aggregate ASR+T5 WER ≤ ASR-only WER − 0.010 absolute,
   AND lower in **at least 6 of 8 folds**.
2. **Helps more than it harms.** Aggregate improved:worsened ≥ **3:1**, and
   improved ≥ **5%** of erroneous test rows (experiment 1: 1:6, 1.1%).
3. **Preserves correct input.** ≥ **98%** of already-correct test rows returned
   unchanged (experiment 1: 96.8%).
4. **Does not damage control speech.** Control WER degrades by ≤ **0.002**
   absolute and control preservation ≥ **99%** (experiment 1: +0.0072, 95.7%).
5. **Reported separately, not pooled:** unseen-speaker (all 683) and
   unseen-prompt (55). Unseen-prompt WER must not be worse than ASR-only.
6. **Abstention check on LOW.** On the 119 LOW-repairability test rows, report
   how often the model rewrites rather than leaves alone. Rewriting these is a
   failure mode: there is no textual evidence to recover the target.

## 7. Cached outputs reused (no recomputation)

| artifact | status |
|---|---|
| `asr_cache.jsonl` (877 medium.en predictions) | **reused in full** |
| `asr_predictions.csv`, `repair_pairs.csv` | reused; only fold labels are re-derived |
| `metadata.csv` (16,552 rows, parsed speakers) | reused |
| `speaker_difficulty.csv` | reused — it defines the fold order |
| `asr_model_comparison.csv` / `asr_selection.json` | reused; medium.en stays fixed |
| `mic_comparison_summary.json` | reused; arrayMic stays out |
| `prompt_frequency.csv`, `dataset_summary.json` | reused |
| control set (194 sentences) | reused unchanged for every fold |

New compute is 8 × (t5-small train + evaluate) ≈ 20–25 min. **No audio is
transcribed again.**

## 8. Caveat on the ordering signal

Fold order comes from ASR difficulty computed over all 8 speakers, including
each fold's test speaker. That is information from the test speaker's audio
influencing the design. It is independent of the repair model and of any target
text, and under LOSO there is no test set being *selected* — every speaker is
tested regardless — so it cannot manufacture a favourable test set. It is noted
here rather than hidden.
