"""Steps 14-19 -- ASR alone vs ASR + T5 on held-out speakers.

    python scripts/t5_evaluate.py

Writes results/t5_small/{test_predictions.csv,evaluation.json,evaluation_report.md}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jiwer
import pandas as pd
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import normalize_text  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
LT_DATA = PROJECT / "data" / "large_torgo"
MODEL_DIR = PROJECT / "models" / "revoice-t5-small"
OUT = PROJECT / "results" / "t5_small"
OUT.mkdir(parents=True, exist_ok=True)

PREFIX = "repair speech: "
MAX_LENGTH = 128
NUM_BEAMS = 4
BATCH_SIZE = 8


def pick_device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


@torch.no_grad()
def repair(model, tokenizer, texts: list[str], device) -> list[str]:
    model.eval()
    out: list[str] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [PREFIX + t for t in texts[i:i + BATCH_SIZE]]
        encoded = tokenizer(batch, max_length=MAX_LENGTH, padding=True,
                            truncation=True, return_tensors="pt").to(device)
        generated = model.generate(**encoded, max_length=MAX_LENGTH,
                                   num_beams=NUM_BEAMS, early_stopping=True)
        out.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return out


def wer_of(references: list[str], hypotheses: list[str]) -> float:
    if not references:
        return float("nan")
    return jiwer.process_words(references,
                               [h if h.strip() else "*" for h in hypotheses]).wer


def per_row_wer(reference: str, hypothesis: str) -> float:
    if not reference.strip():
        return float("nan")
    return jiwer.wer(reference, hypothesis if hypothesis.strip() else "*")


def block(name: str, frame: pd.DataFrame) -> dict:
    """ASR-alone vs ASR+T5 for one subset, plus the improved/worsened split."""
    if frame.empty:
        return {"subset": name, "n": 0}
    refs = list(frame["ground_truth_normalized"])
    asr = list(frame["asr_transcript_normalized"])
    t5 = list(frame["t5_normalized"])
    return {
        "subset": name,
        "n": len(frame),
        "asr_wer": wer_of(refs, asr),
        "asr_exact_match": float((frame["ground_truth_normalized"]
                                  == frame["asr_transcript_normalized"]).mean()),
        "t5_wer": wer_of(refs, t5),
        "t5_exact_match": float((frame["ground_truth_normalized"]
                                 == frame["t5_normalized"]).mean()),
        "improved": int((frame["outcome"] == "IMPROVED").sum()),
        "unchanged": int((frame["outcome"] == "UNCHANGED").sum()),
        "worsened": int((frame["outcome"] == "WORSENED").sum()),
    }


def main() -> None:
    if not (MODEL_DIR / "config.json").exists():
        sys.exit(f"No trained model at {MODEL_DIR}. Run scripts/t5_train.py first.")

    pairs = pd.read_csv(LT_DATA / "repair_pairs.csv", keep_default_na=False, na_values=[""])
    for column in ("repair_input", "repair_target", "asr_transcript_normalized",
                   "ground_truth_normalized"):
        pairs[column] = pairs[column].fillna("")

    evaluated = pairs[pairs["split"].isin(["test", "control_test"])].reset_index(drop=True)
    device = pick_device()
    print(f"device: {device} | evaluating {len(evaluated)} rows")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)

    raw = repair(model, tokenizer, list(evaluated["asr_transcript_normalized"]), device)
    evaluated["t5_output"] = raw
    evaluated["t5_normalized"] = [normalize_text(t) for t in raw]

    evaluated["asr_wer_row"] = [per_row_wer(r, h) for r, h in
                                zip(evaluated["ground_truth_normalized"],
                                    evaluated["asr_transcript_normalized"])]
    evaluated["t5_wer_row"] = [per_row_wer(r, h) for r, h in
                               zip(evaluated["ground_truth_normalized"],
                                   evaluated["t5_normalized"])]
    # IMPROVED / WORSENED are decided on per-sample WER before vs after repair.
    evaluated["outcome"] = [
        "IMPROVED" if after < before - 1e-9 else
        "WORSENED" if after > before + 1e-9 else "UNCHANGED"
        for before, after in zip(evaluated["asr_wer_row"], evaluated["t5_wer_row"])]
    evaluated["t5_changed_text"] = (evaluated["t5_normalized"]
                                    != evaluated["asr_transcript_normalized"])

    evaluated.to_csv(OUT / "test_predictions.csv", index=False)

    test = evaluated[evaluated["split"] == "test"]
    control = evaluated[evaluated["split"] == "control_test"]
    unseen = test[test["unseen_prompt_test"] == True]  # noqa: E712
    seen = test[test["unseen_prompt_test"] != True]  # noqa: E712

    # --- correct-ASR preservation (the key safety metric) -------------------
    def preservation(frame: pd.DataFrame) -> dict:
        correct = frame[frame["ground_truth_normalized"]
                        == frame["asr_transcript_normalized"]]
        if correct.empty:
            return {"correct_asr_examples": 0}
        preserved = int((correct["t5_normalized"] == correct["asr_transcript_normalized"]).sum())
        return {
            "correct_asr_examples": int(len(correct)),
            "preserved_unchanged": preserved,
            "unnecessarily_modified": int(len(correct)) - preserved,
            "preservation_rate": preserved / len(correct),
        }

    report = {
        "asr_model": (pairs["asr_model"].dropna().iloc[0]
                      if pairs["asr_model"].notna().any() else None),
        "t5_model_dir": str(MODEL_DIR),
        "test_speakers": sorted(test["speaker_id"].unique().tolist()),
        "control_speakers": sorted(control["speaker_id"].unique().tolist()),
        "main_test": block("test (unseen speakers)", test),
        "seen_prompt_test": block("test, prompt seen in training", seen),
        "unseen_prompt_test": block("test, prompt NOT in training", unseen),
        "control_test": block("control speakers (safety)", control),
        "correct_asr_preservation_test": preservation(test),
        "correct_asr_preservation_control": preservation(control),
        "control_asr_correct_changed_pct": (
            float((control[control["ground_truth_normalized"]
                           == control["asr_transcript_normalized"]]["t5_changed_text"]).mean())
            if not control.empty else None),
    }

    main_block = report["main_test"]
    if main_block["n"]:
        report["main_test"]["wer_absolute_change"] = main_block["t5_wer"] - main_block["asr_wer"]
        report["main_test"]["wer_relative_change"] = (
            (main_block["t5_wer"] - main_block["asr_wer"]) / main_block["asr_wer"]
            if main_block["asr_wer"] else None)

    (OUT / "evaluation.json").write_text(json.dumps(report, indent=2, default=str))
    write_report(report, evaluated, test, control, unseen)

    print(json.dumps({k: v for k, v in report.items() if k != "t5_model_dir"},
                     indent=2, default=str))
    print(f"\nWrote {OUT / 'evaluation.json'}, {OUT / 'evaluation_report.md'}, "
          f"{OUT / 'test_predictions.csv'}")


def example_lines(frame: pd.DataFrame, limit: int) -> str:
    lines = []
    for _, row in frame.head(limit).iterrows():
        lines.append(
            f"\n`{row['sample_id']}` ({row['speaker_id']}, WER "
            f"{row['asr_wer_row']:.2f} -> {row['t5_wer_row']:.2f})\n\n"
            f"```\nGround truth:\n\"{row['ground_truth_normalized']}\"\n\n"
            f"ASR:\n\"{row['asr_transcript_normalized'] or '<empty>'}\"\n\n"
            f"T5:\n\"{row['t5_normalized'] or '<empty>'}\"\n```\n")
    return "".join(lines) if lines else "\n_none_\n"


def write_report(report: dict, evaluated, test, control, unseen) -> None:
    m = report["main_test"]
    lines = [
        "# Revoice repair model — first experiment (t5-small)\n",
        f"\nASR: `{report['asr_model']}` · repair model: `google-t5/t5-small`\n",
        f"\nTest speakers (never trained on): {report['test_speakers']}\n",
        "\n## Main result — held-out dysarthric speakers\n\n",
        "| pipeline | n | WER | exact match |\n|---|---|---|---|\n",
        f"| ASR alone | {m['n']} | {m['asr_wer']:.4f} | {m['asr_exact_match'] * 100:.2f}% |\n",
        f"| ASR + T5 | {m['n']} | {m['t5_wer']:.4f} | {m['t5_exact_match'] * 100:.2f}% |\n",
        f"\nAbsolute WER change: **{m.get('wer_absolute_change', float('nan')):+.4f}**  \n",
        f"Relative WER change: **{(m.get('wer_relative_change') or 0) * 100:+.2f}%**\n",
        "\n## Per-utterance effect of the repair model\n\n",
        "| outcome | count | % |\n|---|---|---|\n",
    ]
    for key in ("improved", "unchanged", "worsened"):
        lines.append(f"| {key.title()} | {m[key]} | {m[key] / m['n'] * 100:.1f}% |\n")

    p = report["correct_asr_preservation_test"]
    lines += ["\n## Correct-input preservation (safety)\n\n"]
    if p.get("correct_asr_examples"):
        lines += [
            f"- correct ASR examples in test: **{p['correct_asr_examples']}**\n",
            f"- preserved unchanged: **{p['preserved_unchanged']}**\n",
            f"- unnecessarily modified: **{p['unnecessarily_modified']}**\n",
            f"- **preservation rate: {p['preservation_rate'] * 100:.1f}%**\n",
        ]
    else:
        lines.append("\n_No test sentence was transcribed correctly by ASR, so preservation "
                     "cannot be measured on this subset._\n")

    u = report["unseen_prompt_test"]
    lines += ["\n## Unseen-prompt test (generalisation)\n\n"]
    if u.get("n"):
        lines += [
            f"- n = **{u['n']}** (test speakers whose prompt never appears in training)\n",
            f"- ASR WER {u['asr_wer']:.4f} -> ASR + T5 WER {u['t5_wer']:.4f}\n",
            f"- improved {u['improved']}, unchanged {u['unchanged']}, worsened {u['worsened']}\n",
            "\n> This subset is very small; treat it as a smoke test for prompt "
            "memorisation, not as a reliable estimate.\n",
        ]
    else:
        lines.append("\n_Empty._\n")

    c = report["control_test"]
    lines += ["\n## Control speech (secondary safety)\n\n"]
    if c.get("n"):
        pc = report["correct_asr_preservation_control"]
        lines += [
            f"- n = **{c['n']}** sentences from control speakers "
            f"{report['control_speakers']}, never trained on\n",
            f"- ASR WER {c['asr_wer']:.4f} -> ASR + T5 WER {c['t5_wer']:.4f}\n",
            f"- improved {c['improved']}, unchanged {c['unchanged']}, worsened {c['worsened']}\n",
        ]
        if pc.get("correct_asr_examples"):
            lines.append(
                f"- of {pc['correct_asr_examples']} already-correct control sentences, "
                f"T5 changed **{pc['unnecessarily_modified']}** "
                f"({(1 - pc['preservation_rate']) * 100:.1f}%)\n")

    lines += ["\n## Examples\n"]
    improved = test[test["outcome"] == "IMPROVED"].sort_values(
        "t5_wer_row").assign(gain=lambda d: d["asr_wer_row"] - d["t5_wer_row"]
                             ).sort_values("gain", ascending=False)
    unchanged_correct = test[(test["outcome"] == "UNCHANGED")
                             & (test["ground_truth_normalized"]
                                == test["asr_transcript_normalized"])]
    unchanged_any = test[test["outcome"] == "UNCHANGED"]
    worsened = test[test["outcome"] == "WORSENED"].assign(
        loss=lambda d: d["t5_wer_row"] - d["asr_wer_row"]).sort_values("loss", ascending=False)

    lines += [f"\n### Successful repairs ({len(improved)} total)\n",
              example_lines(improved, 10),
              f"\n### Unchanged ({len(unchanged_any)} total, "
              f"{len(unchanged_correct)} of them already correct)\n",
              example_lines(unchanged_correct if not unchanged_correct.empty
                            else unchanged_any, 5),
              f"\n### Regressions ({len(worsened)} total)\n",
              example_lines(worsened, 10)]

    (OUT / "evaluation_report.md").write_text("".join(lines))


if __name__ == "__main__":
    main()
