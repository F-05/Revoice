"""E02: MBR / medoid candidate selection on the frozen hybrid H1-H5 pool."""
import json, sys
from pathlib import Path
import pandas as pd, jiwer
RESEARCH = Path.home()/"Desktop/revoice-model-training"
sys.path.insert(0, str(RESEARCH/"scripts"))
RW = Path.home()/"Downloads/revoice-overnight-research-20260830"

pairs = pd.read_csv(RESEARCH/"data/large_torgo/repair_pairs.csv", keep_default_na=False, na_values=[""])
pairs = pairs[pairs.speaker_group=="dysarthric"].set_index("sample_id")
nbest = {}
for l in (RESEARCH/"data/large_torgo/nbest/nbest_cache.jsonl").read_text().splitlines():
    if l.strip():
        r = json.loads(l); nbest[r["sample_id"]] = [h["processed"] for h in r["hypotheses"]]

def rw(a,b): return jiwer.wer(a, b if b.strip() else "*")
rows=[]
for sid, row in pairs.iterrows():
    if sid not in nbest: continue
    a0 = row["asr_transcript_normalized"] or ""
    hyps = ([a0] if a0 else []) + [h for h in nbest[sid] if h and h != a0]
    hyps = hyps[:5]
    ref = row["ground_truth_normalized"]
    if len(hyps) == 1:
        pick = hyps[0]
    else:
        # expected edit distance to the rest of the list (uniform weights)
        costs = [sum(rw(h2, h) for h2 in hyps if h2 != h)/ (len(hyps)-1) for h in hyps]
        # tie-break toward A0 (index 0) via stable argmin
        pick = hyps[costs.index(min(costs))]
    rows.append({"sid": sid, "speaker": row["speaker_id"], "ref": ref, "a0": a0, "mbr": pick})
df = pd.DataFrame(rows)
def cw(refs, hyps): return jiwer.process_words(list(refs), [h if h.strip() else "*" for h in hyps]).wer
res = {"n": len(df), "a0_wer": cw(df.ref, df.a0), "mbr_wer": cw(df.ref, df.mbr),
       "changed": int((df.mbr != df.a0).sum()),
       "per_speaker": {s: {"a0": cw(g.ref, g.a0), "mbr": cw(g.ref, g.mbr)} for s, g in df.groupby("speaker")}}
print(json.dumps(res, indent=1))
(RW/"results/e02_mbr.json").write_text(json.dumps(res, indent=1))
