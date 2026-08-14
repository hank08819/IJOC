"""The same measurement, on the thirty-eight real chains.

Nothing about the procedure changes from the fab. The policy commits one stage
at a time in topological order, choosing that stage's outbound service time; the
label is what the proved optimum chose there; the read-out is the placement that
results, scored by the model's own cost; a chain counts when that cost is no
worse than what the firm's own rule achieves on the same chain.

A budget here is a labelled decision seen, and a step is taken every BATCH of
them, so the budget axis measures training rather than the number of chains that
happen to be in the file. Budgets past one pass over the pool are further passes
over it, which is what training on a fixed corpus means.
"""
import json, pickle, sys, time
import numpy as np, torch
import torch.nn as nn
from gsm import cost_of_plan
from labels import order_stages
from sweep import Policy, feats

WIDTHS = (1, 2, 4, 8, 16, 32)
T_GRID = tuple(4 * 2 ** k for k in range(11))   # 4 to 4,096
SEEDS = (0, 1, 2, 3, 4)   # the locator needs five; see algorithm.tex
BATCH, LR = 32, 3e-3


def flatten(data):
    """Every labelled decision in the training chains, as one pool."""
    return [(d["net"], i, SI, hi, star)
            for d in data for (i, SI, hi, star) in d["labels"] if hi > 0]


def train_on(pool, width, n_labels, seed):
    torch.manual_seed(seed)
    pol = Policy(width)
    opt = torch.optim.Adam(pol.parameters(), lr=LR)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))
    used, loss, k = 0, 0.0, 0
    while used < n_labels:
        net, i, SI, hi, star = pool[order[used % len(pool)]]
        lg = pol.scores(net, i, SI, hi)
        loss = loss + nn.functional.cross_entropy(
            lg.unsqueeze(0), torch.tensor([min(star, hi)]))
        used += 1; k += 1
        if k == BATCH:
            opt.zero_grad(); (loss / k).backward(); opt.step()
            loss, k = 0.0, 0
        if used % len(pool) == 0:
            order = rng.permutation(len(pool))
    if k:
        opt.zero_grad(); (loss / k).backward(); opt.step()
    return pol, used


def score(pol, held):
    """Fraction of held-out chains the policy places no worse than the rule."""
    wins = []
    for d in held:
        net = d["net"]; S = [0] * net.n
        with torch.no_grad():
            for i in order_stages(net):
                SI = max([S[j] for j in net.pred[i]], default=0)
                hi = SI + int(net.tau[i])
                if i in net.demand:
                    hi = min(hi, net.quoted[i])
                S[i] = int(pol.scores(net, i, SI, hi).argmax()) if hi > 0 else 0
        wins.append(cost_of_plan(net, S) <= d["floor"] * (1 + 1e-9))
    return float(np.mean(wins))


def sigma0(held, widths=WIDTHS, seeds=SEEDS):
    """The read-out's success rate with nothing trained.

    One definition, used by the sweep and by the locator alike: an untrained
    program of each width, one per seed, scored the same way a trained one is.
    The seeds are set, because an unseeded draw gives a different number every
    time it is asked and this one is quoted in the paper.
    """
    out = []
    for w in widths:
        for sd in seeds:
            torch.manual_seed(10_000 + sd)
            out.append(score(Policy(w), held))
    return float(np.mean(out))


def split(data, seed=0):
    """Alternate by size, so both halves span eight to two thousand stages."""
    ix = np.argsort([d["net"].n for d in data])
    return [data[i] for i in ix[0::2]], [data[i] for i in ix[1::2]]


if __name__ == "__main__":
    data = pickle.load(open("results/labels_real.pkl", "rb"))
    train, held = split(data)
    pool = flatten(train)
    print(f"{len(data)} real chains, {sum(d['proved'] for d in data)} proved "
          f"optimal.  {len(train)} trained on ({len(pool)} labelled decisions), "
          f"{len(held)} scored.", flush=True)

    t0 = time.time()
    s0 = sigma0(held)
    print(f"untrained, averaged over the widths: sigma_0 = {s0:.2f} "
          f"[{time.time()-t0:.0f} s]\n", flush=True)
    if s0 >= 0.5:
        print("no critical scale at this problem size: the read-out already "
              "succeeds having spent nothing.", flush=True)
        sys.exit(0)

    print(f"{'width':>6} {'P':>8}  "
          + " ".join(f"{t}".rjust(6) for t in T_GRID), flush=True)
    rows = []
    for w in WIDTHS:
        cells = []
        for T in T_GRID:
            s = []
            for sd in SEEDS:
                pol, used = train_on(pool, w, T, sd)
                v = score(pol, held)
                s.append(v)
                rows.append(dict(width=w, P=sum(p.numel()
                                                for p in pol.parameters()),
                                 T=used, seed=sd, sigma=v))
                json.dump(rows, open("results/sweep_real.json", "w"))
            cells.append(f"{np.mean(s):6.2f}")
        P = [r["P"] for r in rows if r["width"] == w][0]
        print(f"{w:>6} {P:>8,}  " + " ".join(cells)
              + f"   [{(time.time()-t0)/60:.0f} min]", flush=True)
