"""The check that decides whether the grid is in the right place.

A scan is worth running only if the thing it is scanning for is inside it. What
the class asks to see is a scale below which the method fails and above which it
works, so both ends have to be present: the smallest program at the largest
budget should fail, and the largest program at the same budget should succeed.
If they behave the same the grid is in the wrong place and the whole scan buys a
figure with no transition in it.

Success is not fixed to one tolerance here. Every instance's distance to its
proved optimum is kept, and the fraction inside a tolerance is printed across a
range of them, because the tolerance that shows a transition is not known before
the numbers exist. What is known is the bar: the best of the ordinary dispatch
rules lands 2 to 9 per cent above the optimum on these instances, so a tolerance
far above that would be passing marks for doing nothing.
"""
import json
import sys
import time

import numpy as np

from efficiency import RULES, dispatch
from smt2020 import load_routes
from train import BATCH, Pool, gaps_of, n_params, score, train

EPS = (0.02, 0.05, 0.08, 0.10, 0.15, 0.20)


def rule_floor(pool):
    """Where the ordinary dispatch rules land, as the bar to clear."""
    out = []
    for inst, opt in pool.items:
        out.append((min(dispatch(inst, r) for r in RULES) - opt) / opt)
    return np.array(out)


def gate(T=51_200, widths=(16, 64), seed=0, pool_size=24, out="gate.jsonl"):
    routes = load_routes("LVHM")
    pool = Pool(routes, 4, 50, pool_size, seed=99)
    floor = rule_floor(pool)
    hdr = "  ".join(f"s{e:g}".rjust(6) for e in EPS)
    print(f"pool {len(pool)} instances, optima proved", flush=True)
    print(f"dispatch floor: mean {floor.mean():.3f}   "
          + "  ".join(f"{score(floor, e):6.2f}" for e in EPS), flush=True)
    print(f"\n{'what':>16} {'P':>8} {'gap':>7}  {hdr}", flush=True)

    rows = []
    for tag, d, steps in [("untrained", widths[-1], 0)] + \
                         [(f"d={w}", w, T // BATCH) for w in widths]:
        t0 = time.time()
        net = train("sparse", d, 2, steps, seed, routes)
        g = gaps_of(net, pool, seed=seed)
        rows.append(dict(tag=tag, d_model=d, P=n_params(net), T=steps * BATCH,
                         steps=steps, seed=seed, gaps=g.round(5).tolist()))
        print(f"{tag:>16} {n_params(net):>8,} {g.mean():>7.3f}  "
              + "  ".join(f"{score(g, e):6.2f}" for e in EPS)
              + f"   {time.time()-t0:.0f}s", flush=True)

    with open(out, "w") as f:
        f.write(json.dumps(dict(floor=floor.round(5).tolist(), eps=list(EPS)))
                + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # The grid is in the right place only if some tolerance separates the ends.
    small, large = rows[1], rows[2]
    spread = [(e, score(np.array(large["gaps"]), e)
               - score(np.array(small["gaps"]), e)) for e in EPS]
    best = max(spread, key=lambda p: p[1])
    print(f"\nlargest separation between the two ends: {best[1]:+.2f} "
          f"at tolerance {best[0]:g}", flush=True)
    if best[1] < 0.3:
        print("The ends do not separate. The grid is in the wrong place; "
              "moving T or the widths comes before running the scan.", flush=True)
    else:
        print("The ends separate. The scan has something to find.", flush=True)


if __name__ == "__main__":
    gate(T=int(sys.argv[1]) if len(sys.argv) > 1 else 51_200)
