"""Oracle analysis of expanded candidate pools (E01 downstream)."""
import json, sys
from pathlib import Path
import pandas as pd, jiwer
RESEARCH = Path.home()/"Desktop/revoice-model-training"
RW = Path.home()/"Downloads/revoice-overnight-research-20260830"

pairs = pd.read_csv(RESEARCH/"data/large_torgo/repair_pairs.csv", keep_default_na=False, na_values=[""])
pairs = pairs[pairs.speaker_group=="dysarthric"].set_index("sample_id")
frozen = {}
for l in (RESEARCH/"data/large_torgo/nbest/nbest_cache.jsonl").read_text().splitlines():
    if l.strip():
        r = json.loads(l); frozen[r["sample_id"]] = [h["processed"] for h in r["hypotheses"]]

def load_pool(name):
    p = RW/f"results/pool_{name}.jsonl"
    out = {}
    if p.exists():
        for l in p.read_text().splitlines():
            if l.strip():
                r = json.loads(l); out[r["sample_id"]] = [h["norm"] for h in r["hyps"]]
    return out

def rw(a,b): return jiwer.wer(a, b if b.strip() else "*")
def cw(refs,hyps): return jiwer.process_words(list(refs),[h if h.strip() else "*" for h in hyps]).wer

def analyze(pools, cap=None):
    rows=[]
    for sid, row in pairs.iterrows():
        a0 = row["asr_transcript_normalized"] or ""
        cands = ([a0] if a0 else [])
        seen = set(cands)
        for pool in pools:
            for h in pool.get(sid, []):
                if h and h not in seen:
                    seen.add(h); cands.append(h)
        if cap: cands = cands[:cap]
        ref = row["ground_truth_normalized"]
        wers = [rw(ref,c) for c in cands]
        best = min(wers)
        rows.append({"sid":sid,"speaker":row["speaker_id"],"ref":ref,"a0":a0,
                     "n":len(cands),"oracle":cands[wers.index(best)],
                     "ref_in": ref in cands})
    df = pd.DataFrame(rows)
    return {"mean_candidates": float(df.n.mean()),
            "a0_wer": cw(df.ref, df.a0), "oracle_wer": cw(df.ref, df.oracle),
            "ref_in_list_pct": float(df.ref_in.mean()*100),
            "per_speaker_oracle": {s: cw(g.ref,g.oracle) for s,g in df.groupby("speaker")}}, df

if __name__ == "__main__":
    frozen_pool = {k: v for k, v in frozen.items()}
    c1, c2, c3 = load_pool("C1_medium_beam24"), load_pool("C2_medium_sample"), load_pool("C3_turbo_beam12")
    adapted = load_pool("C4_adapted_small")
    combos = {
        "P0_frozen_H1H5": [frozen_pool],
        "P1_+beam24": [frozen_pool, c1],
        "P2_+sampling": [frozen_pool, c1, c2],
        "P3_+turbo": [frozen_pool, c1, c2, c3],
    }
    if adapted:
        combos["P4_+adapted"] = [frozen_pool, c1, c2, c3, adapted]
    out = {}
    for name, pools in combos.items():
        if any(len(p)==0 for p in pools): 
            print(f"{name}: pool incomplete, skipping"); continue
        res,_ = analyze(pools)
        out[name] = res
        print(f"{name}: n̄={res['mean_candidates']:.1f} oracle={res['oracle_wer']:.4f} "
              f"ref-in-list={res['ref_in_list_pct']:.1f}%")
    json.dump(out, open(RW/"results/pool_oracles.json","w"), indent=1)
