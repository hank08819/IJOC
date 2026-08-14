"""Learning to schedule from proved optima, and the scale at which it starts.

The policy is taught, one decision at a time, which operation an optimal
schedule ran next. The teacher is CP-SAT and the instances are whole SMT2020
routes, so the target is an optimum that was proved rather than a heuristic that
was accepted.

Three measurements make this worth doing rather than the reinforcement learning
it replaces. Following the optimum's choices inside the contention set reaches
the optimum exactly, so the restricted action set loses nothing and the ceiling
is zero. The best of the ordinary dispatch rules sits 6.8 per cent above it, so
the target to beat is well clear of the ceiling. And a whole route yields about
94 decisions where more than one operation is in contention, which is what a
labelled example is here.

What the sweep measures is the scale at which the process starts to work: how
many parameters the program needs, and how many labelled decisions it has to see.
"""
import json
import pickle
import sys
import time

import numpy as np
import torch

from efficiency import RULES, dispatch
from fab import makespan_of
from policy import encode
from policy_sparse import CLIP, ActiveScheduler
from smt2020 import cut_instance, load_routes

WIDTHS = (2, 4, 8, 16, 32)
T_GRID = (50, 150, 450, 1350, 4000)     # labelled decisions consumed
SEEDS = (0, 1, 2)
K_READOUT = 64
THRESHOLD = 0.5
LR = 1e-3
N_TRAIN = 60                            # instances the labels come from


def optimal_order(inst, tl=90.0):
    """An optimal schedule, as the order its operations start in."""
    from ortools.sat.python import cp_model
    jobs = inst.as_tuples()
    horizon = sum(d for j in jobs for _, d in j)
    m = cp_model.CpModel()
    S, E = {}, {}
    per = {k: [] for k in range(inst.n_machines)}
    for j, ops in enumerate(jobs):
        prev = None
        for k, (mm, d) in enumerate(ops):
            s = m.NewIntVar(0, horizon, ""); e = m.NewIntVar(0, horizon, "")
            per[mm].append(m.NewIntervalVar(s, d, e, ""))
            S[j, k], E[j, k] = s, e
            if prev is not None:
                m.Add(s >= prev)
            prev = e
    for k in per:
        m.AddNoOverlap(per[k])
    span = m.NewIntVar(0, horizon, "span")
    m.AddMaxEquality(span, [E[j, len(o) - 1] for j, o in enumerate(jobs)])
    m.Minimize(span)
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = tl
    sv.parameters.num_search_workers = 8
    st = sv.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None, False
    starts = {jk: sv.Value(S[jk]) for jk in S}
    order = sorted(starts, key=lambda jk: (starts[jk], jk))
    return order, int(sv.ObjectiveValue()), st == cp_model.OPTIMAL


def labels_from(inst, target):
    """Walk the contention sets and record which operation the optimum took.

    Only steps where more than one operation is in contention are kept: the rest
    are forced and carry no decision to learn.
    """
    rank = {jk: i for i, jk in enumerate(target)}
    lens = np.array([len(r) for r in inst.routes])
    start = np.concatenate([[0], np.cumsum(lens)[:-1]])
    mach = np.concatenate([np.asarray(r) for r in inst.routes])
    dur = np.concatenate([np.asarray(t) for t in inst.times]).astype(float)
    nxt = np.zeros(inst.n_jobs, dtype=np.int64)
    cand = start.copy()
    fm = np.zeros(inst.n_machines); fj = np.zeros(inst.n_jobs)
    out, order = [], []
    for _ in range(inst.n_ops):
        live = np.flatnonzero(nxt < lens)
        rows = cand[live]
        s = np.maximum(fm[mach[rows]], fj[live])
        fin = s + dur[rows]
        a = int(fin.argmin())
        m_star = mach[rows[a]]
        conf = np.flatnonzero((mach[rows] == m_star) & (s < fin[a]))
        pi = int(np.argmin([rank[(int(live[c]), int(nxt[live[c]]))]
                            for c in conf]))
        if len(conf) > 1:
            out.append((rows[conf].copy(), pi))
        j = int(live[conf[pi]]); r = int(cand[j])
        order.append((j, int(nxt[j])))
        fm[mach[r]] = fj[j] = max(fm[mach[r]], fj[j]) + dur[r]
        nxt[j] += 1
        if nxt[j] < lens[j]:
            cand[j] = r + 1
    return order, out


def build_labels(path="labels_4x600.pkl", count=N_TRAIN):
    routes = load_routes("LVHM")
    data, t0 = [], time.time()
    for i in range(count):
        inst = cut_instance(routes, 4, 600, np.random.default_rng(5000 + i))
        tgt, opt, proved = optimal_order(inst)
        if tgt is None:
            continue
        order, lab = labels_from(inst, tgt)
        data.append(dict(inst=inst, opt=opt, proved=proved, labels=lab,
                         imitated=makespan_of(inst, order)))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{count}  {(time.time()-t0)/60:.1f} min", flush=True)
    pickle.dump(data, open(path, "wb"))
    n = float(np.mean([len(d["labels"]) for d in data]))
    gap = float(np.mean([(d["imitated"] - d["opt"]) / d["opt"] for d in data]))
    print(f"{path}: {len(data)} instances, {sum(d['proved'] for d in data)} "
          f"proved, {n:.0f} decisions each, following the optimum reaches "
          f"{gap:.1%} above it", flush=True)
    return data


def train_on(data, width, n_labels, seed, layers=2):
    """One pass over n_labels decisions, in the order the instances came."""
    torch.manual_seed(seed)
    net = ActiveScheduler(d_model=width, n_heads=min(4, width), n_layers=layers)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    used = 0
    for d in data:
        if used >= n_labels:
            break
        inst = d["inst"]
        x, idx = encode(inst)
        h = net.encode_ops(x, idx, inst)
        K = net.k(h); graph = h.mean(0)
        scale = np.sqrt(h.shape[1])
        last = torch.zeros_like(graph)
        loss, n = 0.0, 0
        for rows, pick in d["labels"]:
            if used >= n_labels:
                break
            sel = torch.tensor(rows, dtype=torch.long)
            ctx = net.ctx(torch.cat([graph, last]))
            lg = CLIP * torch.tanh((K[sel] @ net.q(ctx)) / scale)
            loss = loss + torch.nn.functional.cross_entropy(
                lg.unsqueeze(0), torch.tensor([pick]))
            n += 1; used += 1
            last = h[rows[pick]]
        if n:
            opt.zero_grad(); (loss / n).backward(); opt.step()
    return net, used


def score(net, pool, floor, k=K_READOUT, seed=0):
    rng = np.random.default_rng(seed)
    gaps = []
    for inst, opt in pool:
        m, order = net.readout(*encode(inst), inst, k=k, rng=rng)
        assert abs(makespan_of(inst, order) - m) < 1e-6
        gaps.append((m - opt) / opt)
    gaps = np.array(gaps)
    return float((gaps <= floor).mean()), float(gaps.mean())


def main():
    try:
        data = pickle.load(open("labels_4x600.pkl", "rb"))
        print(f"labels_4x600.pkl: {len(data)} instances", flush=True)
    except FileNotFoundError:
        print("building labels", flush=True)
        data = build_labels()

    pool = pickle.load(open("pool_makespan_LVHM_4x600_60.pkl", "rb"))
    floor = np.array([(min(dispatch(i, r) for r in RULES) - o) / o
                      for i, o in pool])
    print(f"pool {len(pool)}, dispatch rules {floor.mean():.1%} above the "
          f"bound, beaten on {0.0:.0%} before training\n", flush=True)

    rows, t0 = [], time.time()
    hdr = " ".join(f"T={t}".rjust(9) for t in T_GRID)
    print(f"{'width':>6} {'P':>8}  {hdr}", flush=True)
    for w in WIDTHS:
        cells = []
        for T in T_GRID:
            s = []
            for sd in SEEDS:
                net, used = train_on(data, w, T, sd)
                sig, gap = score(net, pool, floor, seed=sd)
                s.append(sig)
                rows.append(dict(width=w, P=sum(p.numel() for p in
                                                net.parameters()),
                                 T=used, seed=sd, sigma=sig, gap=gap))
            cells.append(f"{np.mean(s):9.2f}")
            with open("imitate.jsonl", "w") as f:
                f.write(json.dumps(dict(widths=list(WIDTHS), T=list(T_GRID),
                                        seeds=list(SEEDS),
                                        threshold=THRESHOLD,
                                        floor=floor.round(5).tolist())) + "\n")
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        P = [r["P"] for r in rows if r["width"] == w][0]
        print(f"{w:>6} {P:>8,}  " + " ".join(cells) +
              f"   [{(time.time()-t0)/60:.0f} min]", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "labels":
        build_labels()
    else:
        main()
