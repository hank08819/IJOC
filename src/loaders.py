"""Readers for published scheduling instances.

Two formats. The first is the one every classical job-shop benchmark uses and
has used since the 1960s; the second is the route-and-tool form a fab dataset
comes in. Both produce the same object the solver and the policy already take,
so nothing downstream changes when real instances replace generated ones.

Why real instances matter here. The lattice member generates its chains, and
that is defensible because the HP model is itself a stated abstraction with a
computable optimum. A fab is not an abstraction anyone agreed on, so the
processing times, the tool counts and the degree of re-entrance have to come
from somewhere other than the author's judgement.
"""
import os
import numpy as np


class LoadedInstance:
    """Same interface as the generated FabInstance."""

    def __init__(self, routes, times, n_machines, name=""):
        self.routes, self.times = routes, times
        self.n_jobs = len(routes)
        self.n_machines = n_machines
        self.n_ops = sum(len(r) for r in routes)
        self.name = name
        # How often a job returns to a machine it has already used. One means a
        # plain job shop; a fab runs well above it.
        rev = [len(r) / len(set(r)) for r in routes]
        self.reentrance = float(np.mean(rev))

    def as_tuples(self):
        return [list(zip(r, t)) for r, t in zip(self.routes, self.times)]


def read_orlib(path):
    """The OR-Library job-shop format, used by FT, LA, ABZ, ORB, SWV, TA, YN.

        n_jobs n_machines
        m d m d m d ...        one line per job, machine then duration

    Lines starting with # or + are comments and are skipped, which is how the
    concatenated OR-Library files separate instances.
    """
    nums = []
    for line in open(path):
        line = line.strip()
        if not line or line[0] in "#+":
            continue
        nums.extend(int(float(t)) for t in line.split())
    n_jobs, n_mach = nums[0], nums[1]
    body = nums[2:]
    per = 2 * n_mach
    routes, times = [], []
    for j in range(n_jobs):
        chunk = body[j * per:(j + 1) * per]
        routes.append(chunk[0::2])
        times.append(chunk[1::2])
    return LoadedInstance(routes, times, n_mach, os.path.basename(path))


def read_fab_csv(path):
    """A fab route table.

        job,step,machine,duration

    One row per operation, `step` giving the order inside the job. This is the
    form SMT2020 and the MIMAC testbeds reduce to once tool groups are mapped
    to integers, and it keeps re-entrance because the same machine may appear
    at several steps of one job.
    """
    import csv
    rows = list(csv.DictReader(open(path)))
    need = {"job", "step", "machine", "duration"}
    missing = need - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    jobs = {}
    for r in rows:
        jobs.setdefault(str(r["job"]), []).append(
            (int(r["step"]), int(r["machine"]), int(float(r["duration"]))))
    routes, times = [], []
    for j in sorted(jobs):
        ops = sorted(jobs[j])
        routes.append([m for _, m, _ in ops])
        times.append([d for _, _, d in ops])
    n_mach = max(m for r in routes for m in r) + 1
    return LoadedInstance(routes, times, n_mach, os.path.basename(path))


def describe(inst):
    per_job = [len(r) for r in inst.routes]
    return (f"{inst.name or 'instance'}: {inst.n_jobs} jobs, "
            f"{inst.n_machines} machines, {inst.n_ops} operations, "
            f"{min(per_job)}-{max(per_job)} per job, "
            f"re-entrance {inst.reentrance:.2f}")
