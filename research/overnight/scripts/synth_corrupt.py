"""Fold-safe synthetic ASR-error generation for ranker training (Family E).

For a given LOSO fold: confusion statistics are mined ONLY from the fold's
TRAINING speakers' real (A0, reference) word alignments. Corrupted pseudo
hypothesis lists are then generated ONLY from TRAINING speakers' reference
sentences. Held-out speakers contribute nothing to generation.

Output: synthetic/fold{K}_synth.jsonl with full provenance per item.
"""
import json, random, sys
from pathlib import Path
import pandas as pd

RESEARCH = Path.home()/"Desktop/revoice-model-training"
RW = Path.home()/"Downloads/revoice-overnight-research-20260830"
SEED = 20260830
N_VARIANTS = 4       # pseudo-hypotheses per sentence
N_LISTS = 2          # lists per sentence

def align_pairs(rows):
    """Word-level confusion + deletion/insertion stats from (a0, ref) pairs."""
    sub, dele, ins = {}, {}, {}
    for _, r in rows.iterrows():
        a, b = r["asr_transcript_normalized"].split(), r["ground_truth_normalized"].split()
        # simple LCS-ish alignment via difflib
        import difflib
        sm = difflib.SequenceMatcher(a=b, b=a)   # ref -> hyp
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "replace" and (i2-i1) == (j2-j1):
                for rw_, hw in zip(b[i1:i2], a[j1:j2]):
                    sub.setdefault(rw_, []).append(hw)
            elif tag == "delete":
                for w in b[i1:i2]: dele[w] = dele.get(w, 0)+1
            elif tag == "insert":
                for w in a[j1:j2]: ins[w] = ins.get(w, 0)+1
    return sub, dele, ins

def corrupt(words, sub, dele, ins_words, rng, strength):
    out = []
    for w in words:
        r = rng.random()
        if w in sub and r < strength:            # learned confusion
            out.append(rng.choice(sub[w]))
        elif w in dele and r < strength*0.5:     # learned deletion
            continue
        elif r < strength*0.15 and out:          # repetition (dysarthric ASR artifact)
            out.append(out[-1]); out.append(w)
        else:
            out.append(w)
    if ins_words and rng.random() < strength*0.4:
        out.insert(rng.randrange(len(out)+1), rng.choice(ins_words))
    return " ".join(out)

def main(fold):
    folds = pd.read_csv(RESEARCH/"data/large_torgo/loso_folds.csv")
    pairs = pd.read_csv(RESEARCH/"data/large_torgo/repair_pairs.csv",
                        keep_default_na=False, na_values=[""])
    pairs = pairs[pairs.speaker_group=="dysarthric"]
    fr = folds[folds["fold"]==fold]
    train_ids = set(fr[fr.role=="train"]["sample_id"])
    train_rows = pairs[pairs.sample_id.isin(train_ids)]
    sub, dele, ins = align_pairs(train_rows)
    ins_words = [w for w, c in ins.items() if c >= 2]
    rng = random.Random(SEED + fold)
    out_path = RW/f"synthetic/fold{fold}_synth.jsonl"
    n = 0
    with out_path.open("w") as out:
        for _, r in train_rows.iterrows():
            ref = r["ground_truth_normalized"]
            words = ref.split()
            if len(words) < 3: continue
            for k in range(N_LISTS):
                strengths = [0.35, 0.2, 0.5, 0.15][:N_VARIANTS]
                hyps = []
                for s in strengths:
                    h = corrupt(words, sub, dele, ins_words, rng, s)
                    if h and h not in hyps: hyps.append(h)
                if rng.random() < 0.4:  # sometimes the clean ref is in the list
                    hyps.insert(rng.randrange(len(hyps)+1), ref)
                if len(hyps) < 2: continue
                out.write(json.dumps({
                    "provenance": {"fold": fold, "source_sample": r["sample_id"],
                                   "generator": "synth_corrupt v1", "seed": SEED+fold,
                                   "trained_on": "train-fold real (A0,ref) alignments only"},
                    "ref": ref, "hyps": hyps}) + "\n")
                n += 1
    print(f"fold {fold}: {n} synthetic lists from {len(train_rows)} real train rows")

if __name__ == "__main__":
    for f in (range(8) if len(sys.argv) < 2 else [int(sys.argv[1])]):
        main(f)
