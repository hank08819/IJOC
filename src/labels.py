"""Labels for safety-stock placement: which stage the optimum quotes what.

The policy commits one stage at a time in topological order, choosing that
stage's outbound service time. The label is what the exact solution chose there.
Only stages with more than one feasible choice carry a decision.
"""
import numpy as np
from gsm import Net, solve, cost_of_plan, heuristics


def order_stages(net):
    """Topological order: suppliers before the stages they feed."""
    return list(np.argsort(net.layer, kind="stable"))


def labels_from(net, S_star):
    """Walk the stages; at each, the service time the optimum quoted."""
    out = []
    for i in order_stages(net):
        SI = max([S_star[j] for j in net.pred[i]], default=0)
        hi = SI + int(net.tau[i])                  # cannot promise faster
        if i in net.demand:
            hi = min(hi, net.quoted[i])
        if hi < 1:
            continue
        out.append((i, SI, hi, int(S_star[i])))    # choices are 0..hi
    return out


def build(count=60, n_stages=12, seed=100, tl=30.0, layers=3):
    data = []
    for k in range(count):
        net = Net(n_stages, np.random.default_rng(seed + k), layers=layers)
        opt, S, proved = solve(net, tl)
        if opt is None:
            continue
        lab = labels_from(net, S)
        data.append(dict(net=net, opt=opt, proved=proved, labels=lab,
                         S=S, floor=min(heuristics(net).values())))
    return data


if __name__ == "__main__":
    import pickle, sys, time
    t0 = time.time()
    L = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    d = build(int(sys.argv[1]) if len(sys.argv) > 1 else 60, layers=L)
    pickle.dump(d, open(f"labels_gsm_L{L}.pkl", "wb"))
    n = np.mean([len(x["labels"]) for x in d])
    g = np.array([(x["floor"] - x["opt"]) / x["opt"] for x in d])
    print(f"{len(d)} instances, {sum(x['proved'] for x in d)} proved, "
          f"{n:.0f} decisions each, incumbent {g.mean():.1%} above the optimum, "
          f"{(g<=0.001).mean():.0%} of instances already optimal, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
