# OVERNIGHT_RESEARCH_CANDIDATE — "D4-dev": evidence-fused selector (E07-frozenW)

**Development-grade result. NOT integrated. NOT deployed. Fresh-speaker
confirmation required before any production decision.**

## What it is
The deployed constrained-selection architecture, unchanged in shape, with the
ranker upgraded to 16 evidence features — the decisive addition being
**teacher-forced whisper-medium.en sum-NLL per candidate** (a unified acoustic
likelihood), plus GPT-2 NLL, wav2vec2-CTC NLL, and medoid-consensus distance.
Candidate pool = the FROZEN production hybrid H1–H5 (bigger pools rank worse,
E05b/E07-p3W). Output always ∈ H1–H5; hallucination structurally impossible.

## Numbers (nested LOSO over the 683 development rows)
| metric | A0 | D2 (frozen research) | D3-NL (dev) | **D4-dev** |
|---|---|---|---|---|
| WER | 0.3175 | 0.2849 | 0.2853 | **0.2760** |
| exact match | 40.1% | 40.1% | 40.3% | **42.6%** |
| improved/worsened | — | 47/10 | 48/8 | **78/20 (3.9:1)** |
| correct-input preservation | — | 98.91% | 99.64% | **99.27%** |
| unseen-prompt WER | 0.1466 | 0.173 | 0.1466 | **0.1378 (improved)** |
| LOW WER | 1.043 | 0.829 | 0.843 | **0.813 (3 harmful)** |
| unsupported generation | — | 0% | 0% | **0%** |
| oracle capture (frozen pool) | — | 31% | 31% | **39%** |
Per speaker: 7/8 improved vs A0 (M03 0.0242→0.0171, F03 0.133→0.122,
M05→0.249, M02→0.373, M01→0.547, M04→0.620, F01→0.571; F04 +0.002).

## Falsification review (performed 07:05)
- Features audited reference-independent (code: e05_ranker.py feats_for) ✓
- Frozen speaker-disjoint LOSO folds; per-fold scalers; test rows never in
  training ✓ — but the 683 rows are development data (repeatedly inspected),
  so ALL numbers are development-grade by policy.
- wmed/gpt2/w2v scores depend only on (audio, candidate text) — inference-legal ✓
- Ablation attribution: identical run without wmed features = 0.2849 (E05a);
  the gain is the acoustic-likelihood feature, not tuning.
- Known residuals: prompt repetition in TORGO (156 unique prompts) — though the
  unseen-prompt subset improving is evidence against pure memorization; ratio
  3.9:1 sits marginally under the 4:1 product bar; edit precision 65% < 75%.

## Reproduction
    cd ~/Desktop/revoice-model-training
    ./.venv/bin/python ~/Downloads/revoice-overnight-research-20260830/scripts/e05_ranker.py \
        frozenW - nosynth
(Requires results/scores_{gpt2,w2v,wmed}.jsonl in the research workspace —
regenerable via scripts/score_candidates.py and scripts/e06_whisper_rescore.py.)
Predictions: results/best_candidate/e05_frozenW_preds.csv
Safety: results/best_candidate/e07_frozenW_safety.json

## Deployment cost if ever promoted (after fresh-speaker validation)
+ teacher-forced scoring of ≤5 candidates per utterance: ~0.3 s (MPS fp16 HF
medium.en, +3 GB weights) or ~1–2 s CPU; gpt2 (+0.5 GB) and wav2vec2 (+0.4 GB)
optional — the ablation suggests wmed is the load-bearing feature; a
wmed+consensus-only variant should be tested before adding the others.
