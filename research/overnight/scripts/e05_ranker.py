"""E05: constrained ranker over the expanded pool with all evidence families.

Hypothesis (before results): with (a) a bigger candidate pool, (b) LM NLL,
(c) wav2vec2-CTC acoustic NLL, (d) consensus/medoid features, a small listwise
ranker trained per LOSO fold (real + synthetic lists) captures far more oracle
headroom than D3's 31%, pushing dev WER toward the pool oracle.

Output constrained to pool members. Nested LOSO; scalers per fold.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, jiwer

RESEARCH = Path.home()/"Desktop/revoice-model-training"
sys.path.insert(0, str(RESEARCH/"scripts"))
RW = Path.home()/"Downloads/revoice-overnight-research-20260830"
SEED = 20260830

def rw(a,b): return jiwer.wer(a, b if b.strip() else "*")
def cw(refs,hyps): return jiwer.process_words(list(refs),[h if h.strip() else "*" for h in hyps]).wer

def load_jsonl_map(path, key, val):
    out = {}
    p = Path(path)
    if p.exists():
        for l in p.read_text().splitlines():
            if l.strip():
                r = json.loads(l); out[r[key]] = r[val]
    return out

def build_dataset(use_pools):
    pairs = pd.read_csv(RESEARCH/"data/large_torgo/repair_pairs.csv", keep_default_na=False, na_values=[""])
    pairs = pairs[pairs.speaker_group=="dysarthric"].set_index("sample_id")
    pools = {}
    def add(sid, h):
        if h: pools.setdefault(sid, [])
        if h and h not in pools[sid]: pools[sid].append(h)
    for sid, row in pairs.iterrows():
        add(sid, row["asr_transcript_normalized"])
    for l in (RESEARCH/"data/large_torgo/nbest/nbest_cache.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            for h in r["hypotheses"]: add(r["sample_id"], h["processed"])
    for name in use_pools:
        for l in (RW/f"results/pool_{name}.jsonl").read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                for h in r["hyps"]: add(r["sample_id"], h["norm"])
    gpt2 = load_jsonl_map(RW/"results/scores_gpt2.jsonl", "sample_id", "scores")
    w2v = load_jsonl_map(RW/"results/scores_w2v.jsonl", "sample_id", "scores")
    wmed = load_jsonl_map(RW/"results/scores_wmed.jsonl", "sample_id", "scores")
    from transformers import WhisperTokenizer
    _tok = WhisperTokenizer.from_pretrained("openai/whisper-medium.en")
    _nt = {}
    def sum_nll(scores, c):
        v = scores.get(c)
        if v is None: return None
        if c not in _nt: _nt[c] = max(len(_tok(c).input_ids), 1)
        return v * _nt[c]
    utts = []
    for sid, row in pairs.iterrows():
        cands = pools.get(sid, [""])[:24]
        ref = row["ground_truth_normalized"]
        a0 = row["asr_transcript_normalized"] or ""
        wers = [rw(ref, c) for c in cands]
        wm = wmed.get(sid, {})
        utts.append({"sid": sid, "speaker": row["speaker_id"], "ref": ref, "a0": a0,
                     "cands": cands, "label": int(np.argmin(wers)),
                     "gpt2": gpt2.get(sid, {}), "w2v": w2v.get(sid, {}),
                     "wmed": {c: sum_nll(wm, c) for c in cands}})
    return utts

def feats_for(u):
    cands, a0 = u["cands"], u["a0"]
    sets = [set(c.split()) for c in cands]
    from collections import Counter
    support = Counter(w for s in sets for w in s)
    a0set = set(a0.split())
    g, w = u["gpt2"], u["w2v"]
    wm = u.get("wmed", {})
    mvals = [wm.get(c) for c in cands]
    mv = [x for x in mvals if x is not None]
    mmu, msd = (np.mean(mv), np.std(mv)+1e-6) if mv else (0,1)
    gvals = [g.get(c) for c in cands]; wvals = [w.get(c) for c in cands]
    gv = [x for x in gvals if x is not None]; wv = [x for x in wvals if x is not None]
    gmu, gsd = (np.mean(gv), np.std(gv)+1e-6) if gv else (0,1)
    wmu, wsd = (np.mean(wv), np.std(wv)+1e-6) if wv else (0,1)
    rows = []
    for i, c in enumerate(cands):
        words = c.split(); cs = sets[i]
        others = [sets[j] for j in range(len(cands)) if j != i]
        f1 = [2*len(cs&o)/(len(cs)+len(o)) if (cs and o) else 0.0 for o in others]
        changed = [x for x in words if x not in a0set]
        rows.append([
            float(c == a0), float(i), float(len(words)),
            float(np.mean(f1)) if f1 else 1.0, float(max(f1)) if f1 else 1.0,
            float(np.mean([support[x]>=3 for x in words])) if words else 0.0,
            float(np.mean([support[x]>=2 for x in changed])) if changed else 1.0,
            float(any(support[x]==1 for x in changed)),
            float((not changed) and bool(a0set-cs) and c != a0),
            ((gvals[i]-gmu)/gsd) if gvals[i] is not None else 0.0,
            float(gvals[i] is None),
            ((wvals[i]-wmu)/wsd) if wvals[i] is not None else 0.0,
            float(wvals[i] is None),
            float(np.mean([rw(o_c, c) for o_c in cands if o_c != c])) if len(cands)>1 else 0.0,  # medoid dist
            ((mvals[i]-mmu)/msd) if mvals[i] is not None else 0.0,
            float(mvals[i] is None),
        ])
    return np.array(rows, dtype=np.float64)

FEAT_NAMES = ["is_a0","rank","n_words","mean_f1","max_f1","word_sup3","changed_sup2",
              "novel_word","deletion_only","gpt2_z","gpt2_missing","w2v_z","w2v_missing","medoid_dist",
              "wmed_sum_z","wmed_missing"]

def synth_utts(fold):
    out = []
    p = RW/f"synthetic/fold{fold}_synth.jsonl"
    if not p.exists(): return out
    for l in p.read_text().splitlines():
        if not l.strip(): continue
        r = json.loads(l)
        cands = r["hyps"][:24]
        wers = [rw(r["ref"], c) for c in cands]
        out.append({"sid": None, "speaker": None, "ref": r["ref"],
                    "a0": cands[0], "cands": cands, "label": int(np.argmin(wers)),
                    "gpt2": {}, "w2v": {}, "wmed": {}})
    return out

def train_fold(train_u, val_u, dim, epochs=200):
    torch.manual_seed(SEED)
    X = np.vstack([feats_for(u) for u in train_u])
    mu, sd = X.mean(0), X.std(0); sd[sd<1e-9] = 1.0
    def tensors(us): return [(torch.tensor((feats_for(u)-mu)/sd, dtype=torch.float32), u["label"]) for u in us]
    tr, va = tensors(train_u), tensors(val_u)
    net = torch.nn.Sequential(torch.nn.Linear(dim,16), torch.nn.Tanh(), torch.nn.Linear(16,1))
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-3)
    best, best_state = 1e9, None
    import copy
    for ep in range(epochs):
        opt.zero_grad()
        loss = torch.stack([torch.nn.functional.cross_entropy(net(x).squeeze(-1).unsqueeze(0), torch.tensor([y])) for x,y in tr]).mean()
        loss.backward(); opt.step()
        if (ep+1) % 10 == 0:
            with torch.no_grad():
                v = torch.stack([torch.nn.functional.cross_entropy(net(x).squeeze(-1).unsqueeze(0), torch.tensor([y])) for x,y in va]).mean().item()
            if v < best: best, best_state = v, copy.deepcopy(net.state_dict())
    if best_state: net.load_state_dict(best_state)
    return net, mu, sd

def main(use_pools, use_synth, tag):
    utts = build_dataset(use_pools)
    folds = pd.read_csv(RESEARCH/"data/large_torgo/loso_folds.csv")
    by_id = {u["sid"]: u for u in utts}
    dim = len(FEAT_NAMES)
    preds = []
    for fold in sorted(folds["fold"].unique()):
        fr = folds[folds["fold"]==fold]
        get = lambda role: [by_id[s] for s in fr[fr.role==role]["sample_id"] if s in by_id]
        tr, va, te = get("train"), get("validation"), get("test")
        if use_synth: tr = tr + synth_utts(fold)
        net, mu, sd = train_fold(tr, va, dim)
        for u in te:
            x = torch.tensor((feats_for(u)-mu)/sd, dtype=torch.float32)
            with torch.no_grad(): pick = int(net(x).squeeze(-1).argmax())
            assert u["cands"][pick] in u["cands"]
            preds.append({"sid": u["sid"], "speaker": u["speaker"], "ref": u["ref"],
                          "a0": u["a0"], "picked": u["cands"][pick],
                          "oracle": u["cands"][u["label"]], "n_cands": len(u["cands"])})
        print(f"  [{tag}] fold {fold} done", flush=True)
    df = pd.DataFrame(preds)
    res = {"tag": tag, "n": len(df),
           "a0_wer": cw(df.ref, df.a0), "ranker_wer": cw(df.ref, df.picked),
           "oracle_wer": cw(df.ref, df.oracle),
           "mean_cands": float(df.n_cands.mean()),
           "changed": int((df.picked != df.a0).sum()),
           "per_speaker": {s: {"a0": cw(g.ref,g.a0), "ranker": cw(g.ref,g.picked)}
                           for s,g in df.groupby("speaker")}}
    df.to_csv(RW/f"results/e05_{tag}_preds.csv", index=False)
    json.dump(res, open(RW/f"results/e05_{tag}.json","w"), indent=1)
    print(json.dumps({k:v for k,v in res.items() if k!="per_speaker"}, indent=1), flush=True)

if __name__ == "__main__":
    tag = sys.argv[1]; pools = sys.argv[2].split(",") if sys.argv[2] != "-" else []
    main(pools, sys.argv[3] == "synth", tag)
