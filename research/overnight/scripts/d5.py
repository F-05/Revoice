"""D5: Stage-1 feature ablation + Stage-2 hierarchical shortlists + Stage-3 ranker.

All from cached artifacts. Shortlists are reference-independent. Predeclared
designs S1/S2/S3 only. Dev-grade evaluation on the frozen LOSO folds.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, jiwer
RESEARCH = Path.home()/"Desktop/revoice-model-training"
sys.path.insert(0, str(RESEARCH/"scripts"))
RW = Path.home()/"Downloads/revoice-overnight-research-20260830"
sys.path.insert(0, str(RW/"scripts"))
import e05_ranker as R  # reuse dataset/feature machinery (16 feats incl wmed)

SEED = 20260830
IDX = {n:i for i,n in enumerate(R.FEAT_NAMES)}
ABLATIONS = {
 "B_list+wmed": [n for n in R.FEAT_NAMES if not n.startswith(("gpt2","w2v"))],
 "C_wmed+consensus": ["is_a0","mean_f1","max_f1","word_sup3","changed_sup2",
                      "novel_word","deletion_only","medoid_dist","wmed_sum_z","wmed_missing"],
 "D_wmed_only": ["is_a0","wmed_sum_z","wmed_missing"],
}

def rw(a,b): return jiwer.wer(a, b if b.strip() else "*")
def cw(refs,hyps): return jiwer.process_words(list(refs),[h if h.strip() else "*" for h in hyps]).wer

def run_ranker(utts, keep_names, tag):
    keep = [IDX[n] for n in keep_names]
    folds = pd.read_csv(RESEARCH/"data/large_torgo/loso_folds.csv")
    by_id = {u["sid"]: u for u in utts}
    import copy
    preds=[]
    for fold in sorted(folds["fold"].unique()):
        fr = folds[folds["fold"]==fold]
        get = lambda role: [by_id[s] for s in fr[fr.role==role]["sample_id"] if s in by_id]
        tr, va, te = get("train"), get("validation"), get("test")
        torch.manual_seed(SEED)
        X = np.vstack([R.feats_for(u)[:,keep] for u in tr])
        mu, sd = X.mean(0), X.std(0); sd[sd<1e-9]=1.0
        def tensors(us): return [(torch.tensor((R.feats_for(u)[:,keep]-mu)/sd, dtype=torch.float32), u["label"]) for u in us]
        trt, vat = tensors(tr), tensors(va)
        net = torch.nn.Sequential(torch.nn.Linear(len(keep),16), torch.nn.Tanh(), torch.nn.Linear(16,1))
        opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-3)
        best, best_state = 1e9, None
        for ep in range(200):
            opt.zero_grad()
            loss = torch.stack([torch.nn.functional.cross_entropy(net(x).squeeze(-1).unsqueeze(0), torch.tensor([y])) for x,y in trt]).mean()
            loss.backward(); opt.step()
            if (ep+1)%10==0:
                with torch.no_grad():
                    v = torch.stack([torch.nn.functional.cross_entropy(net(x).squeeze(-1).unsqueeze(0), torch.tensor([y])) for x,y in vat]).mean().item()
                if v<best: best,best_state=v,copy.deepcopy(net.state_dict())
        if best_state: net.load_state_dict(best_state)
        for u in te:
            x = torch.tensor((R.feats_for(u)[:,keep]-mu)/sd, dtype=torch.float32)
            with torch.no_grad(): pick = int(net(x).squeeze(-1).argmax())
            picked = u["cands"][pick]
            assert picked in u["cands"]
            preds.append({"sid":u["sid"],"speaker":u["speaker"],"ref":u["ref"],"a0":u["a0"],
                          "picked":picked,"oracle":u["cands"][u["label"]]})
    df = pd.DataFrame(preds)
    df["aw"]=[rw(a,b) for a,b in zip(df.ref,df.a0)]
    df["pw"]=[rw(a,b) for a,b in zip(df.ref,df.picked)]
    sw = df[df.picked!=df.a0]; corr = df[df.ref==df.a0]
    imp=int((df.pw<df.aw-1e-9).sum()); wor=int((df.pw>df.aw+1e-9).sum())
    res = {"tag":tag,"wer":cw(df.ref,df.picked),"oracle":cw(df.ref,df.oracle),
           "improved":imp,"worsened":wor,
           "edit_precision": float((sw.pw<sw.aw-1e-9).mean()) if len(sw) else None,
           "preservation": float((corr.picked==corr.a0).mean()),
           "switch_rate": float((df.picked!=df.a0).mean()),
           "exact": float((df.ref==df.picked).mean()),
           "per_speaker": {s: cw(g.ref,g.picked) for s,g in df.groupby("speaker")}}
    return res, df

# ---------- shortlists (reference-independent) ----------
def build_shortlists():
    utts = R.build_dataset(["C1_medium_beam24","C2_medium_sample","C3_turbo_beam12"])
    out = {"S1":[], "S2":[], "S3":[]}
    for u in utts:
        cands, a0 = u["cands"], u["a0"]
        wm = u["wmed"]
        def nll(c): v = wm.get(c); return v if v is not None else 9e9
        non_a0 = [c for c in cands if c != a0]
        by_ac = sorted(non_a0, key=nll)
        s1 = ([a0] if a0 else []) + by_ac[:4]
        # consensus pick among remaining
        rest2 = [c for c in by_ac[3:]]
        def medoid(c): return np.mean([rw(o,c) for o in cands if o!=c]) if len(cands)>1 else 1
        s2 = ([a0] if a0 else []) + by_ac[:3] + (sorted(rest2, key=medoid)[:1] if rest2 else [])
        # diversity: candidate maximizing new-word coverage among acoustically top-8
        base = set(w for c in ([a0]+by_ac[:3]) for w in c.split())
        pool8 = by_ac[3:8]
        s3 = ([a0] if a0 else []) + by_ac[:3] + \
             (sorted(pool8, key=lambda c: -len(set(c.split())-base))[:1] if pool8 else [])
        for k, sl in (("S1",s1),("S2",s2),("S3",s3)):
            v = dict(u); v = {**u, "cands": sl[:5]}
            wers = [rw(u["ref"], c) for c in v["cands"]]
            v["label"] = int(np.argmin(wers))
            out[k].append(v)
    return utts, out

def oracle_of(utts):
    refs = [u["ref"] for u in utts]
    picks = [u["cands"][u["label"]] for u in utts]
    ref_in = np.mean([u["ref"] in u["cands"] for u in utts])
    n = np.mean([len(u["cands"]) for u in utts])
    return cw(refs,picks), float(ref_in*100), float(n)

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "ablate":
        utts = R.build_dataset([])  # frozen pool
        for tag, names in ABLATIONS.items():
            res,_ = run_ranker(utts, names, tag)
            print(json.dumps({k:v for k,v in res.items() if k!="per_speaker"}))
            json.dump(res, open(RW/f"results/d5_ablate_{tag}.json","w"), indent=1)
    elif mode == "shortlists":
        full, sls = build_shortlists()
        o, ri, nn = oracle_of(full)
        print(f"P3 full: oracle {o:.4f} ref-in {ri:.1f}% n̄={nn:.1f}")
        for k, us in sls.items():
            o, ri, nn = oracle_of(us)
            print(f"{k}: oracle {o:.4f} ref-in {ri:.1f}% n̄={nn:.1f}")
            json.dump({"oracle":o,"ref_in":ri,"mean_n":nn}, open(RW/f"results/d5_{k}_oracle.json","w"))
    elif mode.startswith("final"):
        _, sls = build_shortlists()
        sl = sys.argv[2]; feat = sys.argv[3]
        names = R.FEAT_NAMES if feat=="A" else ABLATIONS[feat]
        res, df = run_ranker(sls[sl], names, f"D5_{sl}_{feat}")
        df.to_csv(RW/"results/d5_final_preds.csv", index=False)
        json.dump(res, open(RW/"results/d5_final.json","w"), indent=1)
        print(json.dumps(res, indent=1))
