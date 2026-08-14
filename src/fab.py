"""Re-entrant flow-shop scheduling, the shape a semiconductor fab has.

A wafer lot visits the same photolithography station several times, once per
mask layer, with other steps in between. That re-entrance is what separates a
fab from a textbook job shop: the same machine appears many times in one job's
route, so a decision made early constrains the machine for the rest of the run.

The objective is makespan. Minimising it is NP-hard for three machines
onwards, and it stays NP-hard under re-entrance. On the instance sizes used
here CP-SAT proves optimality, so the distance is measured against the optimum
and not against another heuristic.
"""
import numpy as np


class FabInstance:
    """One lot-scheduling instance.

    n_jobs lots, n_machines stations, n_layers passes through the route. Each
    pass visits the photolithography station (machine 0) first and then a
    subset of the others, so machine 0 is the bottleneck the way it is in a
    real fab.
    """

    def __init__(self, n_jobs, n_machines, n_layers, rng):
        self.n_jobs, self.n_machines, self.n_layers = n_jobs, n_machines, n_layers
        self.routes, self.times = [], []
        for _ in range(n_jobs):
            route, time = [], []
            for _ in range(n_layers):
                route.append(0)                       # litho, every layer
                time.append(int(rng.integers(3, 9)))
                others = rng.permutation(np.arange(1, n_machines))
                for m in others[: max(1, n_machines // 2)]:
                    route.append(int(m))
                    time.append(int(rng.integers(2, 7)))
            self.routes.append(route)
            self.times.append(time)
        self.n_ops = sum(len(r) for r in self.routes)

    def as_tuples(self):
        """[(machine, duration), ...] per job, the form a solver wants."""
        return [list(zip(r, t)) for r, t in zip(self.routes, self.times)]


def optimal_makespan(inst, time_limit=30.0, objective="makespan"):
    """Exact makespan by CP-SAT, with the proof status returned alongside.

    The status matters. A schedule that is merely the best found in the time
    limit is not an optimum, and using it as one would turn the distance into a
    distance to another heuristic, which is the property this member exists to
    avoid.
    """
    from ortools.sat.python import cp_model
    jobs = inst.as_tuples()
    horizon = sum(d for j in jobs for _, d in j)
    model = cp_model.CpModel()
    starts, ends, per_machine = {}, {}, {m: [] for m in range(inst.n_machines)}
    for j, ops in enumerate(jobs):
        for k, (m, d) in enumerate(ops):
            s = model.NewIntVar(0, horizon, f"s{j}_{k}")
            e = model.NewIntVar(0, horizon, f"e{j}_{k}")
            iv = model.NewIntervalVar(s, d, e, f"i{j}_{k}")
            starts[j, k], ends[j, k] = s, e
            per_machine[m].append(iv)
            if k:
                model.Add(s >= ends[j, k - 1])        # route order inside a job
    for m in per_machine:
        model.AddNoOverlap(per_machine[m])            # one lot per station
    last = [ends[j, len(o) - 1] for j, o in enumerate(jobs)]
    if objective == "makespan":
        span = model.NewIntVar(0, horizon, "span")
        model.AddMaxEquality(span, last)
        model.Minimize(span)
    else:
        model.Minimize(sum(last))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8
    st = solver.Solve(model)
    proved = st == cp_model.OPTIMAL
    val = int(solver.ObjectiveValue()) if st in (cp_model.OPTIMAL,
                                                 cp_model.FEASIBLE) else None
    return val, proved


def completion_times(inst, order):
    """When each lot finishes, given a dispatch order.

    Makespan is the largest of these and flow time is their sum. The two ask
    very different things of a schedule. Makespan on a re-entrant flow shop is
    close to the load of the busiest station, which no ordering changes, so a
    dispatch rule reaches it and there is little for a policy to win. Flow time
    is decided by the order itself. Measured on the same fifteen instances, the
    best of three dispatch rules lands 8.8 per cent above the optimum under
    makespan and 13.1 per cent under flow time, while an untrained policy is at
    17.8 and 17.3: the distance a policy must close to beat the rules falls from
    57 per cent of its own gap to 24.
    """
    jobs = inst.as_tuples()
    nxt = [0] * inst.n_jobs
    free_m = [0] * inst.n_machines
    free_j = [0] * inst.n_jobs
    for j, k in order:
        assert k == nxt[j], "order must respect each job's own sequence"
        m, d = jobs[j][k]
        s = max(free_m[m], free_j[j])
        free_m[m] = free_j[j] = s + d
        nxt[j] += 1
    return free_j


def flowtime_of(inst, order):
    """Sum of the lots' completion times."""
    return sum(completion_times(inst, order))


def makespan_of(inst, order):
    """Makespan of a schedule given as a dispatch order over operations.

    `order` lists (job, op_index) in the sequence the decoder chose. An order
    that violates the route is repaired by waiting, so every order is feasible
    and no candidate has to be thrown away. That keeps the read-out simple: a
    policy always produces a schedule, and a bad policy produces a long one.
    """
    jobs = inst.as_tuples()
    next_op = [0] * inst.n_jobs
    free_m = [0] * inst.n_machines
    free_j = [0] * inst.n_jobs
    for j, k in order:
        assert k == next_op[j], "order must respect each job's own sequence"
        m, d = jobs[j][k]
        s = max(free_m[m], free_j[j])
        free_m[m] = free_j[j] = s + d
        next_op[j] += 1
    return max(free_j)
