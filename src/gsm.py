"""Guaranteed-service safety stock, and the reading that comes before training.

Each stage quotes an outbound service time to its customers and is quoted one by
its suppliers. Its net replenishment time is what it must cover with stock:
inbound service plus its own processing minus what it promises outward. Safety
stock cost grows with the square root of that, so quoting a long service time is
cheap for the stage and expensive for whoever is downstream. Choosing the vector
of service times is the guaranteed-service model of Graves and Willems, and on a
general network it is NP-hard.

Service times are integers here, which is how the model is usually solved, and
the cost of each stage is looked up rather than linearised, so the mixed-integer
program is exact for the discretisation.
"""
import sys
import numpy as np
from ortools.sat.python import cp_model

Z = 1.65                                     # 95 per cent service


class Net:
    """A layered network: raw stages feed sub-assemblies feed demand stages."""

    def __init__(self, n_stages, rng, layers=3):
        self.n = n_stages
        self.layer = np.sort(rng.integers(0, layers, n_stages))
        self.tau = rng.integers(1, 8, n_stages)          # processing time
        self.h = np.round(rng.uniform(1, 10, n_stages), 2)   # holding cost
        self.sigma = np.round(rng.uniform(5, 40, n_stages), 1)
        self.pred = [[] for _ in range(n_stages)]
        for i in range(n_stages):
            up = [j for j in range(n_stages) if self.layer[j] == self.layer[i] - 1]
            if up:
                k = int(rng.integers(1, min(3, len(up)) + 1))
                self.pred[i] = list(rng.choice(up, size=k, replace=False))
        self.demand = [i for i in range(n_stages)
                       if self.layer[i] == self.layer.max()]
        self.quoted = {i: int(rng.integers(0, 4)) for i in self.demand}
        self.cap = int(self.tau.sum())

    def cost_of(self, i, net_time):
        return float(self.h[i] * Z * self.sigma[i] * np.sqrt(max(net_time, 0)))


def solve(net, tl=30.0):
    """The exact placement, by mixed-integer program over integer service times."""
    m = cp_model.CpModel()
    SC = 100                                   # cost scale, costs are integers
    # a stage cannot promise faster than nothing nor slower than the longest
    # chain of processing times reaching it; on the real chains that bound is
    # a few dozen where the sum of all processing times is tens of thousands
    bd = getattr(net, "bound", None)
    hi = [int(bd[i]) if bd is not None else net.cap for i in range(net.n)]
    S = [m.NewIntVar(0, hi[i], f"S{i}") for i in range(net.n)]
    # inbound service is bounded by what the suppliers can promise, which is
    # this stage's bound less its own processing; that keeps the table below
    # exactly as long as the stage's own range
    SI = [m.NewIntVar(0, max(0, hi[i] - int(net.tau[i])), f"I{i}")
          for i in range(net.n)]
    total = []
    for i in range(net.n):
        for j in net.pred[i]:
            m.Add(SI[i] >= S[j])
        if not net.pred[i]:
            m.Add(SI[i] == 0)
        nrt = m.NewIntVar(0, hi[i], f"N{i}")
        m.Add(nrt == SI[i] + int(net.tau[i]) - S[i])
        c = m.NewIntVar(0, 10 ** 9, f"C{i}")
        m.AddElement(nrt, [int(round(net.cost_of(i, t) * SC))
                           for t in range(hi[i] + 1)], c)
        total.append(c)
    for i in net.demand:
        m.Add(S[i] <= net.quoted[i])
    m.Minimize(sum(total))
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = tl
    # One worker and a fixed seed, because the labels are the argument of the
    # minimum and not the minimum. Several workers racing return whichever
    # optimal vector finished first, so two runs agree on the cost and disagree
    # on the service times, and everything trained on those labels then differs.
    # Reproducing a table is worth more here than the seconds this costs.
    sv.parameters.num_search_workers = 1
    sv.parameters.random_seed = 0
    st = sv.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None, False
    return (sv.ObjectiveValue() / SC, [sv.Value(v) for v in S],
            st == cp_model.OPTIMAL)


def cost_of_plan(net, S):
    """What a service-time vector costs, repaired to feasibility by waiting."""
    SI = [0] * net.n
    for i in range(net.n):
        SI[i] = max([S[j] for j in net.pred[i]], default=0)
    return sum(net.cost_of(i, SI[i] + net.tau[i] - S[i]) for i in range(net.n))


def heuristics(net):
    """What a firm does now: quote nothing, or quote everything it may."""
    out = {}
    out["hold everywhere"] = cost_of_plan(net, [0] * net.n)
    S = [net.cap] * net.n
    for i in net.demand:
        S[i] = net.quoted[i]
    # a stage cannot promise more than its inbound plus its own processing
    for _ in range(net.n):
        for i in range(net.n):
            SI = max([S[j] for j in net.pred[i]], default=0)
            S[i] = min(S[i], SI + int(net.tau[i]))
    out["quote as late as allowed"] = cost_of_plan(net, S)
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    rng = np.random.default_rng(0)
    print(f"{n} stages, {count} instances\n", flush=True)
    print(f"{'inst':>5} {'optimum':>10} {'proved':>7} "
          f"{'best heuristic':>15} {'gap':>8}", flush=True)
    gaps = []
    for i in range(count):
        net = Net(n, np.random.default_rng(100 + i))
        opt, S, proved = solve(net)
        if opt is None:
            continue
        hb = min(heuristics(net).values())
        gaps.append((hb - opt) / opt)
        print(f"{i:>5} {opt:>10.0f} {str(proved):>7} {hb:>15.0f} "
              f"{gaps[-1]:>7.1%}", flush=True)
    g = np.array(gaps)
    print(f"\nthe incumbent leaves {g.mean():.1%} on the table on average, "
          f"{(g <= 0.001).mean():.0%} of instances at the optimum", flush=True)
    print("sigma_0 for a policy that only had to match the incumbent would be "
          f"{(g <= 0.001).mean():.2f}", flush=True)
