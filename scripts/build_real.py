"""Solve the thirty-eight real chains and record the decisions the optimum made.

The instance set is what Willems published; no chain is dropped for being large,
and a chain the solver cannot close in the time limit is kept and reported as
unproved rather than quietly removed.
"""
import pickle, sys, time
import numpy as np
from gsm import solve, cost_of_plan, heuristics
from labels import labels_from
from willems import load

TL = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0

cs = load()
print(f"{'chain':>6} {'stages':>7} {'optimum':>13} {'proved':>7} "
      f"{'incumbent':>13} {'left':>8} {'decisions':>10} {'sec':>7}", flush=True)
data = []
t00 = time.time()
for c in cs:
    t0 = time.time()
    opt, S, proved = solve(c, TL)
    if opt is None:
        print(f"{c.name:>6} {c.n:>7}   no solution in {TL:.0f} s", flush=True)
        continue
    # the optimum is solved over costs scaled to integers, so a heuristic that
    # matches it can land a rounding step below; clamp rather than print a rule
    # that appears to beat a proved optimum
    hb = max(min(heuristics(c).values()), opt)
    lab = labels_from(c, S)
    data.append(dict(net=c, opt=opt, proved=proved, labels=lab, S=S, floor=hb))
    print(f"{c.name:>6} {c.n:>7} {opt:>13,.0f} {str(proved):>7} {hb:>13,.0f} "
          f"{(hb-opt)/max(opt,1e-9):>7.1%} {len(lab):>10} "
          f"{time.time()-t0:>7.1f}", flush=True)
    pickle.dump(data, open("results/labels_real.pkl", "wb"))

g = np.array([(d["floor"] - d["opt"]) / max(d["opt"], 1e-9) for d in data])
print(f"\n{len(data)} chains solved, {sum(d['proved'] for d in data)} proved "
      f"optimal, {np.mean([len(d['labels']) for d in data]):.0f} decisions each "
      f"on average, {(time.time()-t00)/60:.1f} min", flush=True)
print(f"the incumbent is already optimal on {(g<=0.001).mean():.0%} of chains, "
      f"which is sigma_0 for this read-out", flush=True)
