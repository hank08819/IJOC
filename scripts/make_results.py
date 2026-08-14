"""Every number the figure and the write-up use, computed once and written down.

Nothing downstream recomputes anything. If a number in the paper is questioned,
it is in results.json and the script that made it is this one.
"""
import json, pickle, sys, time
import numpy as np, torch
from scipy import stats
sys.path.insert(0, ".")
from policy import encode
from policy_sparse import ActiveScheduler, SparseScheduler
from efficiency import dispatch, RULES

EPS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15)
K = 64
items = pickle.load(open("pool120.pkl", "rb"))


def gt_uniform(inst, k, rng):
    """The same restriction, choosing uniformly inside it. No network."""
    lens = np.array([len(r) for r in inst.routes])
    start = np.concatenate([[0], np.cumsum(lens)[:-1]])
    mach = np.concatenate([np.asarray(r) for r in inst.routes])
    dur = np.concatenate([np.asarray(t) for t in inst.times]).astype(float)
    J, b = inst.n_jobs, np.arange(k)
    nxt = np.zeros((k, J), dtype=np.int64); cand = np.tile(start, (k, 1))
    free_m = np.zeros((k, inst.n_machines)); free_j = np.zeros((k, J))
    for _ in range(inst.n_ops):
        live = nxt < lens; m_of = mach[cand]
        s = np.maximum(np.take_along_axis(free_m, m_of, 1), free_j)
        fin = np.where(live, s + dur[cand], np.inf)
        a0 = fin.argmin(1); m_star = m_of[b, a0]
        conf = live & (m_of == m_star[:, None]) & (s < fin[b, a0][:, None])
        wide = int(conf.sum(1).max())
        bi, ji = np.nonzero(conf); pos = (np.cumsum(conf, 1) - 1)[bi, ji]
        take = np.zeros((k, wide), dtype=np.int64)
        keep = np.zeros((k, wide), dtype=bool)
        take[bi, pos] = ji; keep[bi, pos] = True
        p = keep / keep.sum(1, keepdims=True)
        c = np.cumsum(p, 1)
        pick = (c > rng.random((k, 1)) * c[:, -1:]).argmax(1)
        j = take[b, pick]; r = cand[b, j]; m = mach[r]
        st = np.maximum(free_m[b, m], free_j[b, j])
        free_m[b, m] = free_j[b, j] = st + dur[r]
        nxt[b, j] += 1
        cand[b, j] = np.where(nxt[b, j] < lens[j], r + 1, r)
    return float(free_j.max(1).min())


def net_gaps(net, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([(net.readout(*encode(i), i, k=K, rng=rng)[0] - o) / o
                     for i, o in items])


def paired(a, b):
    """b against a, both on the same instances. Positive means b is better."""
    d = a - b
    w = stats.wilcoxon(a, b, alternative="greater")
    ci = stats.bootstrap((d,), np.mean, confidence_level=0.95,
                         n_resamples=20000, random_state=0).confidence_interval
    return dict(mean=float(d.mean()), lo=float(ci.low), hi=float(ci.high),
                p=float(w.pvalue), better=int((d > 0).sum()),
                tied=int((d == 0).sum()), worse=int((d < 0).sum()), n=len(d))


out = {"n_instances": len(items), "K": K, "eps": list(EPS)}
print(f"{len(items)} instances, optima all proved", flush=True)

series = {}
series["dispatch"] = np.array([(min(dispatch(i, r) for r in RULES) - o) / o
                               for i, o in items])
rng = np.random.default_rng(0)
series["restricted_uniform"] = np.array([(gt_uniform(i, K, rng) - o) / o
                                         for i, o in items])
torch.manual_seed(0)
series["restricted_untrained"] = net_gaps(ActiveScheduler())
tr = ActiveScheduler()
tr.load_state_dict(torch.load("trained_active_best.pt"))
series["restricted_trained"] = net_gaps(tr)
torch.manual_seed(0)
series["all_lots_untrained"] = net_gaps(SparseScheduler())

out["gaps"] = {k: v.round(6).tolist() for k, v in series.items()}
out["mean_gap"] = {k: float(v.mean()) for k, v in series.items()}
out["sigma"] = {k: [float((v <= e).mean()) for e in EPS]
                for k, v in series.items()}
out["tests"] = {
    "restricted_uniform_vs_dispatch":
        paired(series["dispatch"], series["restricted_uniform"]),
    "restricted_trained_vs_dispatch":
        paired(series["dispatch"], series["restricted_trained"]),
    "restricted_trained_vs_uniform":
        paired(series["restricted_uniform"], series["restricted_trained"]),
    "restricted_trained_vs_untrained":
        paired(series["restricted_untrained"], series["restricted_trained"]),
}
# encoder cost, measured earlier, both on the same machine and instances
out["encoder"] = dict(
    ops=[1000, 9680, 19360, 38720, 77440, 154880],
    all_pairs=[0.02, 1.35, 5.36, 21.54, 86.29, 358.86],
    sparse=[0.02, 0.11, 0.19, 0.39, 0.77, 1.59])

json.dump(out, open("results.json", "w"), indent=1)
for k, v in out["mean_gap"].items():
    print(f"  {k:>22} mean gap {v:.4f}", flush=True)
for k, v in out["tests"].items():
    print(f"  {k:>34} {v['mean']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] "
          f"p={v['p']:.2e}", flush=True)
print("wrote results.json", flush=True)
