"""What a fast schedule is worth, as a ratio.

Being fast is not by itself worth anything. A dispatch rule answers instantly
and answers badly, and an exact solver will pass its answer within a moment of
starting, so the time it saved bought nothing. The ratio below is written so
that this shows up as a low score rather than a high one:

    E = (time the exact solver needs to first reach the policy's makespan)
        / (time the policy needs to produce it)

Quality is held fixed and only the time differs. A policy that answers quickly
and poorly is caught, because the solver reaches a poor answer almost at once
and the numerator collapses. E above one is time actually saved; E at or below
one says to drop the policy and run the solver.

Two things are reported alongside and neither is optional. Past the size where
CP-SAT can be waited out, the numerator is a censored lower bound rather than a
value, and it is written as such. And at every size the policy is put against
the ordinary dispatch rules, because a large E earned by beating nothing is not
worth having.

The solver is given one worker, since the policy runs in one stream, and both
are timed on the same machine.
"""
import time

import numpy as np
from ortools.sat.python import cp_model

from fab import makespan_of

RULES = ("FIFO", "SPT", "MWKR")


def dispatch(inst, rule):
    """A schedule from an ordinary dispatch rule, for the quality floor.

    This is the non-delay schedule builder a fab dispatcher describes: find the
    operation that could start earliest, look at the station it needs, and let
    the rule choose among the lots queued for that station. The rule decides
    which lot goes, which is where these rules differ from one another -- first
    in first out by how long the lot has waited, shortest processing time by the
    step's own duration, most work remaining by what the lot has left.
    """
    jobs = inst.as_tuples()
    nxt = [0] * inst.n_jobs
    free_m = [0] * inst.n_machines
    free_j = [0] * inst.n_jobs
    left = [sum(d for _, d in o) for o in jobs]
    order = []
    for _ in range(inst.n_ops):
        ready = []
        for j, ops in enumerate(jobs):
            if nxt[j] < len(ops):
                m, d = ops[nxt[j]]
                ready.append((max(free_m[m], free_j[j]), m, d, j))
        t_star, m_star = min(ready)[0], min(ready)[1]

        # the lots competing for that station, and what the rule makes of them
        queue = [(s, m, d, j) for s, m, d, j in ready
                 if m == m_star and s <= t_star]
        key = {"FIFO": lambda q: (free_j[q[3]], q[3]),
               "SPT": lambda q: (q[2], q[3]),
               "MWKR": lambda q: (-left[q[3]], q[3])}[rule]
        s, m, d, j = min(queue, key=key)

        order.append((j, nxt[j]))
        free_m[m] = free_j[j] = s + d
        left[j] -= d
        nxt[j] += 1
    return makespan_of(inst, order)


class _Reached(cp_model.CpSolverSolutionCallback):
    """Stops the solver the moment it matches the target, and notes when."""

    def __init__(self, target, t0):
        super().__init__()
        self.target, self.t0, self.at = target, t0, None

    def on_solution_callback(self):
        if self.ObjectiveValue() <= self.target:
            self.at = time.time() - self.t0
            self.StopSearch()


def time_to_match(inst, target, cap=60.0, workers=1):
    """Wall-clock for CP-SAT to first produce a schedule at least as good.

    Returns the time and whether it got there. A run that hits the cap gives a
    lower bound on the time, not a value, and the caller has to say so.
    """
    jobs = inst.as_tuples()
    horizon = sum(d for j in jobs for _, d in j)
    model = cp_model.CpModel()
    ends, per_machine = {}, {m: [] for m in range(inst.n_machines)}
    for j, ops in enumerate(jobs):
        prev = None
        for k, (m, d) in enumerate(ops):
            s = model.NewIntVar(0, horizon, "")
            e = model.NewIntVar(0, horizon, "")
            per_machine[m].append(model.NewIntervalVar(s, d, e, ""))
            if prev is not None:
                model.Add(s >= prev)
            prev, ends[j, k] = e, e
    for m in per_machine:
        model.AddNoOverlap(per_machine[m])
    span = model.NewIntVar(0, horizon, "span")
    model.AddMaxEquality(span, [ends[j, len(o) - 1] for j, o in enumerate(jobs)])
    model.Minimize(span)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = cap
    solver.parameters.num_search_workers = workers
    t0 = time.time()
    cb = _Reached(target, t0)
    solver.Solve(model, cb)
    return (cb.at, True) if cb.at is not None else (time.time() - t0, False)


def efficiency(net, inst, k=64, rng=None, cap=60.0):
    """E for one instance, with everything needed to read it honestly."""
    from policy import encode
    rng = np.random.default_rng(0) if rng is None else rng
    x, idx = encode(inst)

    t0 = time.time()
    m, order = net.readout(x, idx, inst, k=k, rng=rng)
    t_policy = time.time() - t0
    assert abs(makespan_of(inst, order) - m) < 1e-6, "read-out disagrees"

    t_solver, matched = time_to_match(inst, m, cap=cap)
    floor = {r: dispatch(inst, r) for r in RULES}
    return dict(n_ops=inst.n_ops, n_jobs=inst.n_jobs,
                makespan=m, t_policy=t_policy, t_solver=t_solver,
                E=t_solver / t_policy, censored=not matched,
                beats_rules=bool(m <= min(floor.values())),
                **{f"rule_{r}": floor[r] for r in RULES})
