"""The same measurement, on a supply network instead of a fab.

Nothing about the procedure changes. The policy commits one stage at a time in
topological order, choosing that stage's outbound service time; the label is
what the exact solution chose there; the read-out is the placement that results,
scored by the model's own cost; an instance counts when that cost is no worse
than the incumbent heuristic's on the same instance.
"""
import json, pickle, time
import numpy as np, torch
import torch.nn as nn
from gsm import cost_of_plan
from labels import order_stages

WIDTHS=(2,4,8,16,32); T_GRID=(20,60,180,540,1600); SEEDS=(0,1,2)
FEAT=7; LR=1e-3

def feats(net, i, SI, hi):
    return [net.tau[i]/8, net.h[i]/10, net.sigma[i]/40, SI/10, hi/10,
            len(net.pred[i])/3, 1.0 if i in net.demand else 0.0]

class Policy(nn.Module):
    """Scores each allowed service time for the stage being committed."""
    def __init__(self, d):
        super().__init__()
        self.f=nn.Sequential(nn.Linear(FEAT+1,d), nn.ReLU(),
                             nn.Linear(d,d), nn.ReLU(), nn.Linear(d,1))
    def scores(self, net, i, SI, hi):
        base=feats(net,i,SI,hi)
        x=torch.tensor([base+[s/10] for s in range(hi+1)],dtype=torch.float32)
        return self.f(x).squeeze(1)

def train_on(data, width, n_labels, seed):
    torch.manual_seed(seed)
    net_=Policy(width); opt=torch.optim.Adam(net_.parameters(),lr=LR)
    used=0
    for d in data:
        if used>=n_labels: break
        loss=0.; n=0
        for (i,SI,hi,star) in d["labels"]:
            if used>=n_labels: break
            lg=net_.scores(d["net"],i,SI,hi)
            loss=loss+nn.functional.cross_entropy(lg.unsqueeze(0),
                     torch.tensor([min(star,hi)]))
            n+=1; used+=1
        if n:
            opt.zero_grad(); (loss/n).backward(); opt.step()
    return net_, used

def score(policy, pool):
    wins=[]
    for d in pool:
        net=d["net"]; S=[0]*net.n
        with torch.no_grad():
            for i in order_stages(net):
                SI=max([S[j] for j in net.pred[i]],default=0)
                hi=SI+int(net.tau[i])
                if i in net.demand: hi=min(hi,net.quoted[i])
                S[i]=int(policy.scores(net,i,SI,hi).argmax()) if hi>0 else 0
        wins.append(cost_of_plan(net,S) <= d["floor"]+1e-9)
    return float(np.mean(wins))

if __name__=="__main__":
    import sys
    LAB=sys.argv[1] if len(sys.argv)>1 else "labels_gsm_L3.pkl"
    data=pickle.load(open(LAB,"rb"))
    train, pool = data[:30], data[30:]
    print(f"{LAB}: {len(train)} training, {len(pool)} scored, "
          f"{sum(d['proved'] for d in data)}/{len(data)} proved\n", flush=True)
    print(f"{'width':>6} {'P':>8}  "+" ".join(f"T={t}".rjust(8) for t in T_GRID),
          flush=True)
    rows=[]; t0=time.time()
    for w in WIDTHS:
        cells=[]
        for T in T_GRID:
            s=[]
            for sd in SEEDS:
                pol,used=train_on(train,w,T,sd)
                v=score(pol,pool); s.append(v)
                rows.append(dict(width=w,P=sum(p.numel() for p in pol.parameters()),
                                 T=used,seed=sd,sigma=v))
            cells.append(f"{np.mean(s):8.2f}")
            json.dump(rows,open(LAB.replace("labels","sweep").replace(".pkl",".json"),"w"))
        P=[r["P"] for r in rows if r["width"]==w][0]
        print(f"{w:>6} {P:>8,}  "+" ".join(cells)+f"   [{(time.time()-t0)/60:.0f} min]",
              flush=True)
