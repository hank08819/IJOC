"""The thirty-eight real supply chains of Willems (2008), as GSM instances.

Each chain arrives as two sheets: the stages, with the cost of holding a unit
there, the processing time, and -- for the stages that face a customer -- the
demand and the service time promised to that customer; and the arcs. Nothing
here is generated.

Two quantities the sheets do not state have to be derived, and both are derived
the way the model defines them rather than by a convenient stand-in. A stage's
demand variability is that of the customer stages it feeds, added in quadrature
over the demand stages reachable from it. A stage's service time cannot exceed
the longest chain of processing times leading into it, which is the bound the
program is written over; the sum of all processing times is also a bound and is
far looser, which matters when a chain has two thousand stages.
"""
import numpy as np, pandas as pd
from gsm import Z

XL = "data/willems/MSOM-06-038-R2 Data Set in Excel.xls"
SIC = "data/willems/chains_industry.csv"   # exported from the Access version

# The Excel workbook carries the networks but not what they are. The Access
# version of the same data set carries a table naming the industry of each, and
# it is the reason this member is not only a check that the procedure travels:
# two of the thirty-eight are semiconductor supply chains and eight more are
# computer, storage or communications equipment.
ELECTRONICS = {3572, 3577, 3661, 3674}


class Chain:
    """The interface gsm.solve, gsm.cost_of_plan and gsm.heuristics take."""

    def __init__(self, sd, ll, name):
        self.name = name
        stages = [str(x) for x in sd["Stage Name"]]
        self.name_of = stages
        self.n = n = len(stages)
        ix = {s: i for i, s in enumerate(stages)}
        # Processing times are given to fractional precision and are not exact
        # at any number of decimal places (381.0695 is one of them), while the
        # program is written over integer service times. So they are rounded to
        # the nearest whole period, which affects a quarter of the stages and
        # moves 0.9 per cent of the total time in the data set. An earlier
        # version truncated instead, which shortens every fractional time rather
        # than half of them; check_xml.py is what caught it.
        self.tau = np.rint(np.maximum(
            0, np.nan_to_num(sd["stageTime"].values, nan=0))).astype(int)
        self.h = np.maximum(0.01, np.nan_to_num(sd["stageCost"].values, nan=1.0)
                            ).astype(float)
        sig = np.nan_to_num(sd["stdDevDemand"].values, nan=0.0).astype(float)
        self.sigma_own = sig.copy()   # as stated, before propagation

        self.pred = [[] for _ in range(n)]
        self.succ = [[] for _ in range(n)]
        for _, r in ll.iterrows():
            a, b = r["sourceStage"], r["destinationStage"]
            if a in ix and b in ix and ix[a] != ix[b]:
                self.pred[ix[b]].append(ix[a])
                self.succ[ix[a]].append(ix[b])

        self.order = self._topo()
        self.layer = np.zeros(n, dtype=int)
        for i in self.order:
            for j in self.succ[i]:
                self.layer[j] = max(self.layer[j], self.layer[i] + 1)

        # a stage that supplies no one faces the customer
        self.demand = [i for i in range(n) if not self.succ[i]]
        for i in self.demand:                    # a demand stage needs a demand
            if sig[i] <= 0:
                sig[i] = 1.0
        mst = np.nan_to_num(sd["maxServiceTime"].values, nan=0.0)
        self.quoted = {i: int(max(0, mst[i])) for i in self.demand}

        # demand at an interior stage is that of the demand stages it feeds
        reach = [0] * n
        for i in self.demand:
            reach[i] = 1 << i
        for i in reversed(self.order):
            for j in self.succ[i]:
                reach[i] |= reach[j]
        var = sig ** 2
        self.sigma = np.array([
            np.sqrt(sum(var[k] for k in range(n) if reach[i] >> k & 1))
            if reach[i] else max(sig[i], 1.0) for i in range(n)])

        # the longest chain of processing times into a stage bounds its promise
        depth = np.zeros(n, dtype=int)
        for i in self.order:
            base = max([depth[j] for j in self.pred[i]], default=0)
            depth[i] = base + int(self.tau[i])
        self.bound = depth
        self.cap = int(depth.max())

    def _topo(self):
        indeg = np.array([len(p) for p in self.pred])
        stack = [i for i in range(self.n) if indeg[i] == 0]
        out = []
        while stack:
            i = stack.pop()
            out.append(i)
            for j in self.succ[i]:
                indeg[j] -= 1
                if indeg[j] == 0:
                    stack.append(j)
        if len(out) < self.n:                    # a cycle: keep the rest as read
            out += [i for i in range(self.n) if i not in set(out)]
        return out

    def cost_of(self, i, net_time):
        return float(self.h[i] * Z * self.sigma[i] * np.sqrt(max(net_time, 0)))


def industries(path=SIC):
    """SIC code and description per chain, keyed the way the sheets are named."""
    try:
        t = pd.read_csv(path)
    except OSError:
        return {}
    return {f"{int(r['Chain']):02d}": (int(r["SIC Code"]),
                                       str(r["SIC Description"]).strip())
            for _, r in t.iterrows()}


def load(path=XL, max_stages=10 ** 6):
    x = pd.ExcelFile(path)
    out = []
    for sh in x.sheet_names:
        if not sh.endswith("_SD") or f"{sh[:-3]}_LL" not in x.sheet_names:
            continue
        sd = x.parse(sh)
        if "Stage Name" not in sd.columns or len(sd) > max_stages:
            continue
        out.append(Chain(sd, x.parse(f"{sh[:-3]}_LL"), sh[:-3]))
    ind = industries()
    for c in out:
        c.sic, c.industry = ind.get(c.name, (0, "unknown"))
    return out


if __name__ == "__main__":
    import sys
    cs = load(max_stages=int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 6)
    print(f"{'chain':>6} {'stages':>7} {'arcs':>6} {'demand':>7} {'depth':>6} "
          f"{'cap':>6}   industry", flush=True)
    for c in cs:
        print(f"{c.name:>6} {c.n:>7} {sum(len(p) for p in c.pred):>6} "
              f"{len(c.demand):>7} {c.layer.max()+1:>6} {c.cap:>6}   "
              f"{c.industry}", flush=True)
