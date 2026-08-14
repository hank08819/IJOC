"""The budget axis, read off one trajectory per seed.

Training is sequential, so a checkpoint partway through a run is the same
program a run stopped there would have produced. The whole T axis therefore
comes free from one run: only the program size needs a run of its own. That
turns the cost of locating a crossing in T from probes times seeds into seeds.

Instances are 800 operations, eight lots of a hundred steps cut from SMT2020,
chosen because an untrained policy succeeds on none of them and CP-SAT still
certifies its bound to a tenth of a per cent.
"""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, ".")
import pickle
from policy import encode
from efficiency import dispatch, RULES
from smt2020 import load_routes
import train as TR

EPS = (0.02, 0.05, 0.08, 0.10, 0.15)
SEEDS = (0, 1, 2, 3, 4)
T_TOTAL = 8000
WIDTH = 64

TR.BATCH = 64
TR.WARMUP_MIN_EPOCHS = 10 ** 9
TR.EPOCH = 12                      # a checkpoint every 768 instances
STEPS = T_TOTAL // TR.BATCH

routes = load_routes("LVHM")
OUT = sys.argv[2] if len(sys.argv) > 2 else "search_T.jsonl"
POOL = sys.argv[1] if len(sys.argv) > 1 else "pool_8x100.pkl"
items = pickle.load(open(POOL, "rb"))


class P:
    pass


pool = P(); pool.items = items
floor = np.array([(min(dispatch(i, r) for r in RULES) - lb) / lb
                  for i, lb in items])
print(f"800 operations, {len(items)} instances, bound certified to 0.1%",
      flush=True)
print(f"dispatch rules: mean {floor.mean():.3f}  "
      + " ".join(f"s{e:g}={TR.score(floor, e):.2f}" for e in EPS), flush=True)
print(f"\n{'seed':>4} {'instances':>10} {'gap':>7} "
      + " ".join(f"s{e:g}".rjust(6) for e in EPS) + f" {'mins':>6}", flush=True)

log = []
t0 = time.time()
for sd in SEEDS:
    net0 = TR.make_net("active", WIDTH, 2)
    g = TR.gaps_of(net0, pool, seed=sd)
    log.append(dict(seed=sd, instances=0, gap=float(g.mean()),
                    gaps=g.round(5).tolist()))
    print(f"{sd:>4} {0:>10,} {g.mean():>7.3f} "
          + " ".join(f"{TR.score(g, e):6.2f}" for e in EPS)
          + f" {(time.time()-t0)/60:>6.1f}", flush=True)

    def watch(t, net, warm, sd=sd):
        g = TR.gaps_of(net, pool, seed=sd)
        log.append(dict(seed=sd, instances=t * TR.BATCH, gap=float(g.mean()),
                        gaps=g.round(5).tolist()))
        with open(OUT, "w") as f:
            f.write(json.dumps(dict(eps=list(EPS), width=WIDTH,
                                    batch=TR.BATCH, n_ops=800,
                                    floor=floor.round(5).tolist())) + "\n")
            for r in log:
                f.write(json.dumps(r) + "\n")
        print(f"{sd:>4} {t*TR.BATCH:>10,} {g.mean():>7.3f} "
              + " ".join(f"{TR.score(g, e):6.2f}" for e in EPS)
              + f" {(time.time()-t0)/60:>6.1f}", flush=True)

    TR.train("active", WIDTH, 2, STEPS, sd, routes,
             n_lots=8, window=100, watch=watch)
print("\ndone", flush=True)
