# D3 design — consensus-grounded selector (Stage 1 audit + Stage 2 proposal)

Status: **audit complete, design proposed, NOTHING TRAINED.** D1/D2/D2.1 frozen.

## Stage 1 — what the audit found (61 D2 switches: 47 beneficial / 10 harmful / 4 neutral)

**The margin cannot separate good from bad switches — but the list can.**
D2's own margin has AUC 0.353 (weakly informative). The strongest
inference-time discriminators are all consensus/grounding measures
(AUC distance from 0.5, direction: beneficial switches have MORE support):

| feature | benef mean | harm mean | AUC(harm>ben) | discrimination |
|---|---|---|---|---|
| f13 median # hyps agreeing on changed words | 1.64 | 0.70 | 0.224 | **0.776** |
| f08 max overlap with another hypothesis | 0.84 | 0.70 | 0.233 | **0.767** |
| f09 # hyps supporting changed words | 2.34 | 1.30 | 0.230 | **0.770** |
| f04 mean F1 with other hypotheses | 0.65 | 0.54 | 0.282 | 0.718 |
| f10 changed-words attested fraction | 0.87 | 0.69 | 0.293 | 0.707 |
| f11 changed-words unique fraction | 0.13 | 0.31 | 0.707 | 0.707 |
| f01 edit distance to A0 | 7.06 | 5.00 | 0.298 | 0.702 |
| f17 length diff vs A0 | −3.3 | −2.1 | 0.674 | 0.674 |
| f18 introduces novel words | 0.36 | 0.70 | 0.669 | 0.669 |
| f15 within-list-normalized ct2 score | 0.84 | 0.53 | 0.339 | 0.661 |

Full distributions (mean/median/std/quartiles) in `d3_audit.json`; per-switch
values in `d3_audit_features.csv`.

## F03 harmful switches — every one is a recognisable pattern

(Full detail with complete lists in `d3_f03_audit.md`.)

1. **Novel-word inventions** — picked text contains words attested nowhere else
   in the list: "the **prospentative** cupboard", "get a **caracal capsule**
   kick", "**through delta over** disappointing" (that one: attested-fraction
   0.0, unique-fraction 1.0, mean-F1-others 0.22). D2's existing features
   scored these on rank/score; the candidate's changed words having *zero
   corroboration* was invisible to it.
2. **Beam-truncation family** — A0 = "young people participate in athletic
   activities" (correct), picked "young people participate in", from a list
   whose tail is progressively truncated copies. Consensus F1 looks *high*
   (0.795) because truncations overlap heavily — the tell is that the candidate
   is a **strict deletion of A0** (no changed words, only deletions) with high
   margin 0.823. No current feature flags deletion-only candidates.

Matched beneficial switches at the same margins show the mirror profile:
changed words attested in 2–3 other hypotheses, a near-duplicate ally in the
list, no novel words. **Harmful and beneficial high-margin switches are
separable at inference time.** Answer to the Stage-1 question: yes.

## Stage 2 — proposed D3: five features added to the frozen 16-param scorer

Same conditional logit, same training formulation, same hybrid lists, same
outer LOSO, same frozen τ procedure as D2. Model grows 16 → **21 parameters**.

| new feature | exact formula (per candidate i, changed = words of i absent from A0) | audit evidence | why backend-available |
|---|---|---|---|
| `changed_support_count` | median over changed words of #(other hypotheses containing the word); 0 if no changed words and i≠A0 else list size | AUC 0.776 | pure list text |
| `max_ally_overlap` | max over other hypotheses of word-overlap F1 with i | AUC 0.767 | pure list text |
| `novel_word_flag` | 1 if any changed word appears in NO other hypothesis | AUC 0.669; catches "prospentative"/"caracal" | pure list text |
| `deletion_only_flag` | 1 if i has no changed words AND deletes ≥1 word of A0 | F03 truncation case (margin 0.823) | pure list text |
| `score_within_list` | (ct2 score − min list score) / (max − min), 0 for A0 or degenerate spread | AUC 0.661; **within-list normalization is the direct answer to D2.1's lesson that raw margins don't transfer across speakers** | ct2 scores already cached/available |

Rejected as redundant: f10/f11/f12 (existing `disputed_support` covers the
attested-fraction family), f04 (existing `consensus_f1`), f14 list disagreement
(constant across candidates — cancels in the conditional-logit softmax),
f19/f20 (weak: AUC 0.512/0.550), f01–f03/f17 (edit/length family already
present as `edit_to_a0`, `edit_to_a0_norm`, `n_words`, `len_ratio_median`).

Cost: feature computation is string ops over ≤5 short hypotheses — microseconds;
training unchanged (<1 min for 8 folds); inference unchanged (<1 ms).

## Leakage / overfitting — addressed explicitly

The 683 LOSO predictions have now been inspected across D1/D2/D2.1 and this
audit, and the audit itself used test outcomes to rank features. Mitigations,
stated before training:

1. **Conceptual priors first**: "richer consensus features" and within-list
   normalization were proposed *before* this audit (experiment-3 design §
   features; D2.1 report) — the audit narrowed a pre-existing direction, it did
   not invent one.
2. **No test-tuned constants**: the five features are formulas, not thresholds;
   no cutoff is fitted to test outcomes; the training procedure, hyperparams,
   grid, and τ rule stay byte-identical to D2's frozen versions.
3. **Small addition**: 5 features / 5 new parameters, chosen from 20 audited —
   selected for conceptual class coverage (support-count, ally, novelty,
   truncation, normalization), not purely by AUC rank (else f09/f13 twins
   would both enter).
4. **Honest labelling**: D3's LOSO numbers will be reported with an explicit
   optimistic-bias caveat — feature selection saw these test speakers.
5. **Clean confirmation path (optional, needs approval)**: the reserved
   arrayMic channel can serve as a near-fresh evaluation set (same utterances,
   different recordings, never used in any experiment) at the cost of one new
   transcription+N-best pass.

## Frozen success criteria (before training; unchanged after)

aggregate WER < 0.2849 (D2) · correct-input preservation ≥99% · control
preservation ≥99% · unsupported generation = 0% · improved:worsened ≥4:1 ·
edit precision ≥75% · LOW no worse than D2 (WER ≤0.829, harmful ≤1) ·
**unseen-prompt WER ≤ 0.147 (A0)** · ≥7/8 speakers without meaningful
degradation (tol. +0.005). Stretch (non-binding): ≤0.2723 (C).

**STOPPED — awaiting approval to train D3 (21-parameter selector, LOSO).**
