"""Training the active-schedule decoder, against the bar the dispatch rules set.

Untrained, this decoder already lands at 0.029 with 64 samples, against the
rules' 0.032 and the other decoder's 0.082 after 25,600 instances of training.
So the question has changed. It is no longer whether the method reaches a
useful schedule but whether training still buys anything once the choice set is
restricted to the operations actually contending for a station.

The tolerance to watch is 0.02, not 0.05. At 0.05 the untrained policy is
already at 0.75 and there is little room left to see anything move.
"""
import sys, time, json
import numpy as np, torch
sys.path.insert(0, ".")
from smt2020 import load_routes
from efficiency import dispatch, RULES
import train as TR

T_TOTAL = 256_000
EPS = (0.01, 0.02, 0.03, 0.05, 0.08, 0.10)

TR.BATCH = 64
TR.WARMUP_MIN_EPOCHS = 10 ** 9          # exponential baseline throughout
TR.EPOCH = 200                          # a checkpoint every 12,800 instances
STEPS = T_TOTAL // TR.BATCH

routes = load_routes("LVHM")
pool = TR.Pool(routes, 4, 50, 12, seed=99)
floor = np.array([(min(dispatch(i, r) for r in RULES) - o) / o
                  for i, o in pool.items])

print(f"active-schedule decoder, T={T_TOTAL:,} instances, batch {TR.BATCH}",
      flush=True)
print(f"the bar: dispatch rules, mean {floor.mean():.3f}, "
      + " ".join(f"s{e:g}={TR.score(floor, e):.2f}" for e in EPS), flush=True)
hdr = " ".join(f"s{e:g}".rjust(6) for e in EPS)
print(f"\n{'step':>5} {'instances':>10} {'gap':>7} {hdr} {'mins':>6}", flush=True)

best = [1e9]
t0 = time.time()
log = []


def watch(t, net, warm):
    g = TR.gaps_of(net, pool, seed=0)
    m = float(g.mean())
    if m < best[0]:
        best[0] = m
        torch.save(net.state_dict(), "trained_active_best.pt")
    log.append(dict(step=t, instances=t * TR.BATCH, gap=m,
                    gaps=g.round(5).tolist()))
    with open("long_run_gt.jsonl", "w") as f:
        f.write(json.dumps(dict(T=T_TOTAL, batch=TR.BATCH, eps=list(EPS),
                                floor=floor.round(5).tolist())) + "\n")
        for r in log:
            f.write(json.dumps(r) + "\n")
    print(f"{t:>5} {t*TR.BATCH:>10,} {m:>7.3f} "
          + " ".join(f"{TR.score(g, e):6.2f}" for e in EPS)
          + f" {(time.time()-t0)/60:>6.1f}", flush=True)


net0 = TR.make_net("active", 64, 2)
g0 = TR.gaps_of(net0, pool, seed=0)
print(f"{0:>5} {0:>10,} {g0.mean():>7.3f} "
      + " ".join(f"{TR.score(g0, e):6.2f}" for e in EPS) + f" {0.0:>6.1f}",
      flush=True)

TR.train("active", 64, 2, STEPS, 0, routes, watch=watch)
print(f"\nbest mean gap {best[0]:.3f}  (bar {floor.mean():.3f})", flush=True)
