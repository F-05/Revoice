# Overnight evaluation policy (frozen 03:02 AEST before any experiment)

DATA STATUS
- The 683 headMic dysarthric sentences are DEVELOPMENT DATA: they have shaped
  D1/D2/D2.1/D3 and the feature audit. No overnight result on them is a clean
  unseen-speaker number; everything is labelled development-grade.
- Speaker-aware splitting always: the frozen 8-fold LOSO speaker assignment
  (data/large_torgo/loso_folds.csv in the research project) is reused verbatim
  for any trained component. Scalers/models/synthetic recipes fit on training
  folds only. No speaker-ID features, no per-speaker exceptions.
- arrayMic recordings of the same utterances = "alternate-channel robustness
  evaluation" ONLY, never an unseen-speaker test. Inventory result: arrayMic
  headroom exists (7,477 paired recordings) but has no cached A0; a locked
  arrayMic set may be defined AFTER methods freeze, evaluated once, no tuning.
- No genuinely fresh dysarthric speaker data exists locally (TORGO 8 speakers
  all consumed; control speakers consumed for safety metrics). UASpeech etc.
  require licenses not obtainable tonight. CONCLUSION: fresh-speaker
  confirmation remains required for every overnight result.
- Synthetic data: TRAINING ONLY, generated from training-fold information
  only, versioned under synthetic/ with provenance. Never evaluation.

METRICS: WER via jiwer corpus WER on the project's frozen normalize_text; A0
reference = cached production medium.en transcripts (unchanged); oracle = min
per-utterance WER over a candidate pool; per-speaker breakdowns mandatory;
improved/worsened vs A0 per row; unsupported generation must stay 0% for any
selector-style system (output ∈ candidate pool).

DECODING CONFIGS for candidate expansion are FIXED before results (see
experiment registry E01): no per-utterance or per-speaker config selection.

TIME: no new major experiments after 08:00; hard stop ~08:30.
