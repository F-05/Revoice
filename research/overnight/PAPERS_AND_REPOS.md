# Papers and repositories investigated (overnight 2026-08-30)

## ASR adaptation for dysarthric speech (Family F motivation)
1. **CBA-Whisper: Curriculum Learning-Based AdaLoRA Fine-Tuning on Whisper for
   Low-Resource Dysarthric Speech Recognition** — Tan et al., Interspeech 2025.
   https://www.isca-archive.org/interspeech_2025/tan25b_interspeech.html
   Whisper large-v2 + AdaLoRA on SAP+UASpeech+TORGO → 10.51% WER on SAPC Test2.
   Relevance: adaptation, not post-hoc repair, delivers the big WER cuts.
   Code: not released with paper (checked). Borrowed: the direction only.
2. **Adapting Foundation ASR Models to Dysarthric Speech: A Case Study** —
   arXiv:2606.31722. Whisper large-v3: 1.4 h adaptation data → 15.8% WER;
   22.5 h → 10.7%. LoRA-only notably worse than full FT at equal data.
   Relevance: our ~2–3 h of train-speaker audio is in the regime where
   meaningful gains are documented; full FT of a small checkpoint preferred
   over LoRA at this scale. Borrowed: scale expectations + FT-over-LoRA choice.
3. **Self-Training for Whisper, long dysarthric speech** — Wang et al.,
   Interspeech 2025 / arXiv:2506.22810. Relevance: future work (long-form).

## Acoustic rescoring (Family D)
4. **It's Never Too Late: Fusing Acoustic Information into Large Language
   Models for ASR Error Correction** — ICLR 2024.
   https://proceedings.iclr.cc/paper_files/paper/2024/file/0231de0eff264c0639a4c43728b8b55b-Paper-Conference.pdf
   Fuses acoustic evidence into GER; confirms language-only rescoring is
   insufficient. Borrowed: acoustic+linguistic fusion for ranking.
5. **Non-Intrusive ASR Refinement: A Survey** — arXiv:2508.07285. Taxonomy of
   N-best reranking vs generation. Confirms our constrained-selection stance is
   an established, safer line. Borrowed: framing + feature families.
6. Wav2vec2-CTC candidate scoring (multiple sources above): score each
   hypothesis by CTC loss of the candidate text against the audio under an
   independent acoustic model. Implemented tonight as `w2v_ctc_nll`.

## Language rescoring (Family C)
7. **ProGRes: Prompted Generative Rescoring on ASR n-Best** — arXiv:2409.00217.
   LLM rescoring of n-best; we use the selection-only variant (no generation).
8. **BERT-based reranking LMs for ASR** — arXiv:2104.04950. Pseudo-likelihood
   reranking; we implement the cheaper causal-LM (GPT-2) NLL variant tonight.

## Earlier session (verified previously, reused conceptually)
- **GER4Dys** (github.com/morenolaquatra/ger4dys) — no code released.
- **CHSER/GenSEC** (github.com/balaji1312/CHSER) — atypical-speech GEC recipe.
- **FlanEC** (github.com/MorenoLaQuatra/FlanEC) — N-best prompt format.
- **HyPoradise** (arXiv:2309.15701) — beam-then-dedupe N-best construction.

## Licensing / usage
No external code copied tonight; external models used as published weights
(gpt2: MIT-like MIT? — OpenAI/HF model card license: MIT; wav2vec2-base-960h:
Apache-2.0 per model card; faster-whisper models: MIT). No repos cloned —
GER4Dys has no code, and the remaining methods were reimplemented from paper
descriptions in <200 lines each.
