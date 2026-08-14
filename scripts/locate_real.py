"""Algorithm 1, run on the thirty-eight real chains.

The grid in sweep_real.py measured thirty scales to find one boundary. This
measures the boundary directly: one evaluation with nothing trained to ask
whether a boundary can exist, then doubling outwards on both sides until one
width fails and a larger one succeeds, then bisecting between them, then
checking that success stays when the width is doubled again.

The comparison at the end is the point. Both find the same boundary; one pays
for the area and the other for its edge.
"""
import pickle, sys, time
import numpy as np, torch

sys.path.insert(0, "../IJoC-SUPER-P")
from critical_scale import room, decide, bi_extend, narrow, closed
from sweep import Policy
from sweep_real import flatten, split, score, train_on, sigma0
from gsm import cost_of_plan
from labels import order_stages

BUDGET, SEEDS, THRESH = 256, 9, 0.5
RUNS = [0]
SEEN = {}


def untrained_gaps(held, seeds=SEEDS):
    """How far an untrained program lands from the rule, per chain."""
    out = []
    for sd in range(seeds):
        torch.manual_seed(1000 + sd)
        pol = Policy(8)
        for d in held:
            net = d["net"]; S = [0] * net.n
            with torch.no_grad():
                for i in order_stages(net):
                    SI = max([S[j] for j in net.pred[i]], default=0)
                    hi = SI + int(net.tau[i])
                    if i in net.demand:
                        hi = min(hi, net.quoted[i])
                    S[i] = int(pol.scores(net, i, SI, hi).argmax()) if hi > 0 \
                        else 0
            out.append((cost_of_plan(net, S) - d["floor"]) / d["floor"])
    return out


def main():
    data = pickle.load(open("results/labels_real.pkl", "rb"))
    train, held = split(data)
    pool = flatten(train)
    print(f"{len(data)} real chains, all proved optimal.  {len(train)} trained "
          f"on, {len(held)} scored.  Budget fixed at {BUDGET} labelled "
          f"decisions; the axis searched is the program size.\n", flush=True)
    t0 = time.time()

    # Step 0
    s0 = sigma0(held)
    has_room = s0 < THRESH
    print(f"Step 0.  untrained success rate {s0:.2f} "
          f"({'there is room to train' if has_room else 'no boundary here'}), "
          f"and it cost no training at all", flush=True)
    if not has_room:
        return

    def evaluate(width):
        w = max(1, int(round(width)))
        rates = []
        for sd in range(SEEDS):
            pol, _ = train_on(pool, w, BUDGET, sd)
            rates.append(score(pol, held))
            RUNS[0] += 1
        SEEN[w] = rates
        return float(np.mean(rates)), float(np.std(rates)), \
            decide(rates, THRESH)

    def succeeds(width):
        # narrow() and closed() read the verdict, including "unsure"; handing
        # them a boolean makes every comparison false and the bisection quits
        # on its first probe without narrowing anything
        return evaluate(width)[2]

    print("\nStep 1.  extend on both sides until the boundary is enclosed",
          flush=True)
    lo, hi, seen = bi_extend(evaluate, 8, grow=2.0, limit=8,
                             threshold=THRESH, n_seeds=SEEDS)
    if lo is None:
        print("  no bracket: the process saturates without crossing",
              flush=True)
        return
    print(f"  bracketed: width {lo:g} fails, width {hi:g} succeeds",
          flush=True)

    print("\nStep 2.  halve the interval", flush=True)
    lo, hi, _ = narrow(succeeds, lo, hi, ratio=1.3, limit=8)
    print(f"  narrowed to widths {lo:g} to {hi:g}", flush=True)

    print("\nStep 3.  check that success stays above", flush=True)
    up = closed(succeeds, hi, factors=(2.0, 4.0))
    ok = all(v == "yes" for v in up.values())
    for f, v in up.items():
        print(f"  width {hi*f:g}: {v}", flush=True)
    print(f"  {'upward closed' if ok else 'NOT upward closed'}", flush=True)

    P = lambda w: sum(p.numel()
                      for p in Policy(max(1, int(round(w)))).parameters())
    est = np.sqrt(P(lo) * P(hi))
    print(f"\nP_crit is between {P(lo):,} and {P(hi):,} parameters, "
          f"estimate {est:,.0f}, at {BUDGET} labelled decisions.", flush=True)
    print("\nper-seed success rates at each width probed:", flush=True)
    for w in sorted(SEEN):
        print(f"  width {w:>3}  " + "  ".join(f"{r:.2f}" for r in SEEN[w]),
              flush=True)
    print(f"\n{RUNS[0]} training runs at {SEEDS} seeds, "
          f"{(time.time()-t0)/60:.1f} min.  That buys one point of the "
          f"boundary, at one budget.  The grid in sweep_real.py buys the whole "
          f"curve, six program sizes by eleven budgets at five seeds, for "
          f"{6*11*5} runs.", flush=True)


if __name__ == "__main__":
    main()
