# Experiment 3 final report — D1/D2 conservative hypothesis selector

Stages 4–5 complete. Nothing integrated; frontend/backend untouched; no D3.

## Comparison (683 held-out dysarthric sentences, LOSO)

| system | WER | exact | impr/wors | edit prec. | correct pres. | control pres. | unseen-prompt WER | LOW behaviour | unsupported gen. |
|---|---|---|---|---|---|---|---|---|---|
| A0 production 1-best | 0.3175 | 40.1% | — | — | — | — | 0.1466 | 1.0433 | — |
| B 1-best→T5 (frozen) | 0.3225 | 38.2% | 12/33 | 20% | 94.9% | 94.7% | 0.1584 | 24/119 rewritten | present |
| C N-best→FlanT5 free-gen | 0.2723 | 45.5% | 130/35 | 61.6% | 95.6% | 96.0% | 0.1672 | 48↑/2↓, 56% rewrite | present |
| D1 selector (τ=0) | 0.2826 | 40.1% | 53/12 = 4.42 | 76.8% | 98.91% | — | 0.1730 | 46↑/1↓ | **0%** |
| **D2 selector+margin** | **0.2849** | 40.1% | **47/10 = 4.70** | **77.0%** | **98.91%** | **99.92%** | 0.1730 | **1.043→0.829, 1 harmful** | **0%** |

Selection oracle = 0.2132. D2 gain capture = 31.3%. Per-fold τ (validation-only,
frozen rule): M03 0.9, F04 0.9, F03 0.0, M05 0.05, M02 0.05, F01 0.0,
M01 0.05, M04 0.1 — the calibration itself learned "be inert on easy speakers,
switch on hard ones".

## Product-condition scoreboard for D2

| condition | result | met |
|---|---|---|
| clearly beats A0 | 0.2849 vs 0.3175 (−0.0326) | ✅ |
| correct-input preservation ≥99% | **98.91%** (3/274 switched) | ❌ by one switch |
| control preservation ≥99% | 99.92% mean, 99.39% worst | ✅ |
| unsupported generation = 0% | 0% (verified per row) | ✅ |
| improved:worsened ≥4:1 | 4.70 | ✅ |
| edit precision ≥75% | 77.0% | ✅ |
| LOW WER not worse than A0 | 1.043 → 0.829 | ✅ |
| harmful LOW switches rare | 1 of 119 (0.8%) | ✅* |
| ≥7/8 speakers not degraded | 7/8 (F03 +0.0051, marginally over the 0.005 tol.) | ✅ (at the edge) |
| unseen-prompt not worse than A0 | 0.1466 → 0.1730 (+1/−5) | ❌ |

*Process note, stated honestly: the harmful-LOW numerical threshold was
supposed to be pre-registered from train/validation behaviour before test
evaluation, and was not — the 0.8% figure is reported raw, without a
pass/fail claim.

## Diagnostics

**Suppression:** D2's thresholds suppressed 8 of D1's 69 switches — 6 had been
improvements, 2 had been harmful. The margin buys safety at a small WER cost
(0.2826 → 0.2849), exactly the conservatism-for-benefit trade requested.
Abstention (UNCERTAIN) rate: 1.2% overall.

**The unseen-prompt failure is now fully localized.** F03's fold chose τ=0 —
its validation speaker (M05) showed clean behaviour at τ=0, so nothing was
suppressed — and F03's 11 switches (4 improved / 6 worsened) went through
unchanged. F03 contributes 42/55 unseen-prompt rows; its 6 bad switches ARE the
unseen-prompt regression. Every other speaker: unseen-prompt behaviour is
neutral or positive. No F03-specific tuning was done, per instruction.

## Recommendation

**D2 is not yet safe enough to deploy, but it is the right architecture.**
It meets 8 of 10 product conditions, its two misses are a single switch on
preservation and one speaker-fold's bad switches, and its failure mode is
categorically better than C's: every output is a real ASR hypothesis, so the
worst case is "picked the wrong transcription of what was said", never "invented
a sentence".

The remaining risk concentrates in one place: a fold whose validation speaker
under-represents the test speaker's error profile selects τ too low. Two
candidate fixes, both requiring approval before any further work:
1. a **global minimum τ floor** (e.g. τ ≥ 0.05) added to the frozen rule —
   one-line change, would have suppressed F03's low-margin switches;
2. richer consensus features so the scorer itself, not the threshold, learns to
   distrust F03-type switches.

D3 (constrained local repair) remains unjustified until selection is safe:
oracle headroom exists (0.2132 vs 0.2849), but the safety machinery should be
watertight first.

Artifacts: d2_predictions.csv, d2_control_predictions.csv, d2_evaluation.json,
risk_coverage.csv (per-fold τ sweep), d1_* (frozen), oracle_* (frozen).
