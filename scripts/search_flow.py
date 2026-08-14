"""The program-size axis, searched rather than scanned.

Two program sizes share no trajectory, so unlike the budget this axis costs a
run per point and is where the bracketing is worth doing. Algorithm 1 is applied
as written: extend outwards on both sides from a guess until one size clearly
fails and one clearly succeeds, then halve on a log scale, then check that
success stays when the size is doubled again.

Success is the criterion with nothing chosen in it. An instance counts when the
read-out is at least as good as the best of the three ordinary dispatch rules on
that same instance. The rules are on the same instances and were not written by
us, so the bar is not ours to set. It is a demanding bar -- the rules get the
best of three for free while the policy pays for 64 samples -- and being
demanding in that direction is the safe one.
"""
import json
import math
import sys
import time

import numpy as np
import pickle
import torch

from critical_scale import bi_extend, closed, decide, narrow
from efficiency import dispatch, RULES
from policy import encode
from smt2020 import load_routes
import train as TR
TR.SELF_DRAWS = 2      # two schedules per instance; the baseline is the other

T_BUDGET = 3000
SEEDS = (0, 1, 2)
THRESHOLD = 0.5
START_WIDTH = 32.0
LAYERS = 2

TR.BATCH = 64
TR.WARMUP_MIN_EPOCHS = 10 ** 9
TR.EPOCH = 10 ** 9                 # no checkpoints; the run is one point
WATCH_AT = set()                   # each run is a single point on the P axis

routes = load_routes("LVHM")
POOL = sys.argv[1] if len(sys.argv) > 1 else "pool_flowtime_LVHM_8x100_60.pkl"
items = pickle.load(open(POOL, "rb"))


class Pool:
    pass


pool = Pool(); pool.items = items
def _order(inst, rule):
    """The schedule an ordinary dispatch rule builds, as an order."""
    jobs = inst.as_tuples(); nxt = [0] * inst.n_jobs
    fm = [0] * inst.n_machines; fj = [0] * inst.n_jobs
    left = [sum(d for _, d in o) for o in jobs]; order = []
    for _ in range(inst.n_ops):
        ready = [(max(fm[jobs[j][nxt[j]][0]], fj[j]), jobs[j][nxt[j]][0],
                  jobs[j][nxt[j]][1], j)
                 for j in range(inst.n_jobs) if nxt[j] < len(jobs[j])]
        t_, ms = min(ready)[0], min(ready)[1]
        q = [x for x in ready if x[1] == ms and x[0] <= t_]
        key = {"FIFO": lambda z: (fj[z[3]], z[3]),
               "SPT": lambda z: (z[2], z[3]),
               "MWKR": lambda z: (-left[z[3]], z[3])}[rule]
        s_, mm, d, j = min(q, key=key)
        order.append((j, nxt[j])); fm[mm] = fj[j] = s_ + d
        left[j] -= d; nxt[j] += 1
    return order


# The bar is the rules' own cost under the objective in force. It does not use
# the bound, which CP-SAT could not close for flow time, and neither does the
# success criterion: an instance counts when the policy's schedule costs no more
# than the best rule's on that same instance.
floor = np.array([(min(TR.cost_of(i, _order(i, r)) for r in RULES) - lb) / lb
                  for i, lb in items])
log = []
t0 = time.time()


def evaluate(width):
    """Train at this program size for each seed and score against the rules."""
    w = max(4, int(round(width / 4) * 4))          # heads divide the width
    rates = []
    for sd in SEEDS:
        net = TR.train("active", w, LAYERS, T_BUDGET // TR.BATCH, sd, routes,
                       n_lots=8, window=100, watch_at=WATCH_AT)
        g = TR.gaps_of(net, pool, seed=sd)
        rates.append(float((g <= floor).mean()))
        log.append(dict(width=w, P=TR.n_params(net), seed=sd,
                        rate=rates[-1], gaps=g.round(5).tolist()))
        with open("search_P.jsonl", "w") as f:
            f.write(json.dumps(dict(T=T_BUDGET, layers=LAYERS,
                                    threshold=THRESHOLD, n_ops=800,
                                    floor=floor.round(5).tolist())) + "\n")
            for r in log:
                f.write(json.dumps(r) + "\n")
    r = np.array(rates)
    v = decide(r, THRESHOLD)
    print(f"  width {w:>4}  beats the rules on {r.mean()*100:>5.1f}% of "
          f"instances  spread {r.std(ddof=1):.3f}  {v}  "
          f"[{(time.time()-t0)/60:.0f} min]", flush=True)
    return float(r.mean()), float(r.std(ddof=1)), v


if __name__ == "__main__":
    print(f"800 operations, {len(items)} instances, T={T_BUDGET:,} per run, "
          f"{len(SEEDS)} seeds", flush=True)
    print(f"success: the read-out is at least as good as the best of "
          f"{', '.join(RULES)} on that instance\n", flush=True)

    lo, hi, seen = bi_extend(evaluate, START_WIDTH, grow=2.0, limit=5,
                             n_seeds=len(SEEDS))
    if lo is None:
        print("\nNo bracket. Nothing in the range covered separates failure "
              "from success; report the range, not a scale.", flush=True)
        sys.exit(0)

    print(f"\nbracketed between width {lo:g} and {hi:g}", flush=True)
    a, b, _ = narrow(lambda w: evaluate(w)[2], lo, hi, ratio=1.3, limit=3)
    print(f"boundary between {a:g} and {b:g}, estimate "
          f"{math.sqrt(a*b):.0f}", flush=True)
    ab = closed(lambda w: evaluate(w)[2], b, factors=(2.0,))
    print(f"above it: {ab}", flush=True)
