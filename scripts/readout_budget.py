"""How much of the distance to the optimum is closed by sampling more.

The member's read-out is the best of K sampled schedules, and K was set to 64
without a reason. That matters more than a constant usually does. If the success
rate climbs towards one as K grows, then K is a resource the scan has to declare
alongside the program size and the budget, and the reason the success rate
stalls is that the read-out was too small. If it stalls whatever K is, the
instances themselves are the limit and the scan needs different ones.

The trained policy is written to disk first, so the sweep over K runs against
one set of weights and any later question can be asked of the same file.
"""
import json
import sys
import time

import numpy as np
import torch

from efficiency import RULES, dispatch
from policy import encode
from smt2020 import load_routes
from train import BATCH, Pool, n_params, score, train

EPS = (0.02, 0.05, 0.08, 0.10, 0.15, 0.20)
K_GRID = (1, 4, 16, 64, 256, 1024)


def gaps_at_k(net, pool, k, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for inst, opt in pool.items:
        m, _ = net.readout(*encode(inst), inst, k=k, rng=rng)
        out.append((m - opt) / opt)
    return np.array(out)


def main(T=25_600, d_model=64, seed=0, pool_size=24, ckpt="trained_d64.pt"):
    routes = load_routes("LVHM")
    pool = Pool(routes, 4, 50, pool_size, seed=99)
    floor = np.array([(min(dispatch(i, r) for r in RULES) - o) / o
                      for i, o in pool.items])
    print(f"pool {len(pool)}, optima proved. dispatch floor mean "
          f"{floor.mean():.3f}, best {floor.min():.3f}", flush=True)

    t0 = time.time()
    net = train("sparse", d_model, 2, T // BATCH, seed, routes)
    torch.save(net.state_dict(), ckpt)
    print(f"trained P={n_params(net):,} on T={T:,} instances "
          f"in {time.time()-t0:.0f}s -> {ckpt}\n", flush=True)

    hdr = "  ".join(f"s{e:g}".rjust(6) for e in EPS)
    print(f"{'K':>6} {'gap':>7} {'best':>7}  {hdr} {'secs':>7}", flush=True)
    rows = []
    for k in K_GRID:
        t0 = time.time()
        g = gaps_at_k(net, pool, k, seed=seed)
        rows.append(dict(K=k, gaps=g.round(5).tolist()))
        print(f"{k:>6} {g.mean():>7.3f} {g.min():>7.3f}  "
              + "  ".join(f"{score(g, e):6.2f}" for e in EPS)
              + f" {time.time()-t0:>7.1f}", flush=True)

    with open("readout_budget.jsonl", "w") as f:
        f.write(json.dumps(dict(T=T, d_model=d_model, P=n_params(net),
                                eps=list(EPS),
                                floor=floor.round(5).tolist())) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main(T=int(sys.argv[1]) if len(sys.argv) > 1 else 25_600)
