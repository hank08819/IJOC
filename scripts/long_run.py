"""Ten times the training, against the bar the dispatch rules set.

The scan cannot start until one thing is known: whether this rule, on these
instances, ever gets past what an ordinary dispatch rule already does. At 25,600
instances it does not -- mean distance to the proved optimum 0.082 against the
rules' 0.032. That is not yet a verdict, because Kool et al. train on 128 million
instances and this is 0.02 per cent of that.

So the budget goes up ten times and the trajectory is written down, not just the
end point. The bar is drawn from the same twelve instances the policy is scored
on, so the two numbers are comparable.
"""
import sys, time, json
import numpy as np, torch
sys.path.insert(0, ".")
from smt2020 import load_routes
from efficiency import dispatch, RULES
import train as TR

T_TOTAL = 256_000
EPS = (0.02, 0.05, 0.08, 0.10, 0.15, 0.20)

TR.BATCH = 64
TR.WARMUP_MIN_EPOCHS = 10 ** 9          # exponential baseline throughout
STEPS = T_TOTAL // TR.BATCH

routes = load_routes("LVHM")
pool = TR.Pool(routes, 4, 50, 12, seed=99)
floor = np.array([(min(dispatch(i, r) for r in RULES) - o) / o
                  for i, o in pool.items])

print(f"T={T_TOTAL:,} instances, batch {TR.BATCH}, {STEPS:,} steps", flush=True)
print(f"the bar: dispatch rules on the same twelve instances, "
      f"mean {floor.mean():.3f}, "
      + " ".join(f"s{e:g}={TR.score(floor, e):.2f}" for e in EPS), flush=True)
hdr = " ".join(f"s{e:g}".rjust(6) for e in EPS)
print(f"\n{'step':>5} {'instances':>10} {'gap':>7} {hdr} {'past bar':>9} "
      f"{'mins':>6}", flush=True)

best = [1e9, None]
t0 = time.time()
log = []


def watch(t, net, warm):
    g = TR.gaps_of(net, pool, seed=0)
    m = float(g.mean())
    if m < best[0]:
        best[0] = m
        best[1] = {k: v.clone() for k, v in net.state_dict().items()}
        torch.save(best[1], "trained_best.pt")
    log.append(dict(step=t, instances=t * TR.BATCH, gap=m,
                    gaps=g.round(5).tolist()))
    with open("long_run.jsonl", "w") as f:
        f.write(json.dumps(dict(T=T_TOTAL, batch=TR.BATCH, eps=list(EPS),
                                floor=floor.round(5).tolist())) + "\n")
        for r in log:
            f.write(json.dumps(r) + "\n")
    print(f"{t:>5} {t*TR.BATCH:>10,} {m:>7.3f} "
          + " ".join(f"{TR.score(g, e):6.2f}" for e in EPS)
          + f" {'yes' if m < floor.mean() else 'no':>9} "
          f"{(time.time()-t0)/60:>6.1f}", flush=True)


TR.EPOCH = 400                          # a checkpoint every 25,600 instances
TR.train("sparse", 64, 2, STEPS, 0, routes, watch=watch)
print(f"\nbest mean gap {best[0]:.3f}  (bar {floor.mean():.3f}) "
      f"-> trained_best.pt", flush=True)
