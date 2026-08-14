"""Instances cut from SMT2020, a published model of a real semiconductor fab.

SMT2020 is a testbed released by Kopp, Hassoun, Kalir and Monch. It carries ten
product routes of 242 to 583 process steps over 106 tool groups, with the
processing-time distribution given for every step. The routes were built from
fabs rather than from a generator, and their re-entrance -- 3.5 to 5.6 visits
per tool group -- is the property that separates a fab from a job shop and the
one an author would be most likely to get wrong by guessing.

A whole route cannot be scheduled to proven optimality: 583 steps over 106 tool
groups is far past what CP-SAT will close. So instances are cut from the routes
rather than invented. A window of consecutive steps is taken from each of
several products, keeping the tool groups those steps actually use and the
processing times actually given. What survives the cut is the structure: which
station follows which, how often a lot returns to one it has already used, and
how uneven the processing times are.

What is lost is stated rather than hidden. A window is not a fab. Setups,
batching, preventive maintenance, breakdowns and transport are all in SMT2020
and none of them is here, because each would change the objective from makespan
to something a solver could not close at this size.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DATA",
                    "smt2020", "SMT_2020 - Final", "General Data")
DATASETS = {
    "HVLM": ("dataset 1", "SMT_2020_Model_Data_-_HVLM.xlsx"),
    "LVHM": ("dataset 2", "SMT_2020_Model_Data_-_LVHM.xlsx"),
}


class SMTInstance:
    """Same interface the solver and the policy already take."""

    def __init__(self, routes, times, n_machines, meta):
        self.routes, self.times = routes, times
        self.n_jobs = len(routes)
        self.n_machines = n_machines
        self.n_ops = sum(len(r) for r in routes)
        self.meta = meta
        rev = [len(r) / len(set(r)) for r in routes]
        self.reentrance = float(np.mean(rev))

    def as_tuples(self):
        return [list(zip(r, t)) for r, t in zip(self.routes, self.times)]


def load_routes(dataset="LVHM"):
    """Every product route, as a table of step, tool group and mean time."""
    sub, fn = DATASETS[dataset]
    x = pd.ExcelFile(os.path.join(ROOT, sub, fn))
    out = {}
    for s in x.sheet_names:
        if not s.startswith("Route_Product_"):
            continue
        d = x.parse(s)
        d = d[["STEP", "AREA", "TOOLGROUP", "MEAN"]].dropna(subset=["TOOLGROUP"])
        out[s.replace("Route_", "")] = d.sort_values("STEP").reset_index(drop=True)
    return out


def cut_instance(routes, n_jobs, window, rng, dataset="LVHM"):
    """Cut one schedulable instance out of the real routes.

    n_jobs products, `window` consecutive steps from each, starting at a random
    offset. Tool groups are renumbered to the ones the window actually touches,
    so the machine count is whatever those steps need. Times are the published
    means, rounded to whole minutes, since the solver wants integers.

    Past ten lots the same route is drawn more than once, each draw with its own
    starting offset. That is what a fab does: ten products, many lots of each,
    all competing for the same tool groups.
    """
    names = sorted(routes)
    picked = rng.choice(names, size=n_jobs, replace=n_jobs > len(names))
    frames = []
    for nm in picked:
        d = routes[nm]
        start = int(rng.integers(0, max(1, len(d) - window)))
        frames.append(d.iloc[start:start + window])

    tools = sorted({t for f in frames for t in f["TOOLGROUP"]})
    ix = {t: i for i, t in enumerate(tools)}
    rts, tms = [], []
    for f in frames:
        rts.append([ix[t] for t in f["TOOLGROUP"]])
        tms.append([max(1, int(round(float(m)))) for m in f["MEAN"]])
    areas = {ix[t]: a for f in frames
             for t, a in zip(f["TOOLGROUP"], f["AREA"])}
    return SMTInstance(rts, tms, len(tools),
                       dict(dataset=dataset, products=list(picked),
                            window=window, areas=areas))


def describe(inst):
    litho = [m for m, a in inst.meta["areas"].items() if "Litho" in str(a)]
    return (f"{inst.n_jobs} lots x {inst.n_ops//inst.n_jobs} steps, "
            f"{inst.n_machines} tool groups, re-entrance {inst.reentrance:.2f}, "
            f"{len(litho)} lithography groups")
