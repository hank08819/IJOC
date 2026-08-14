"""A pool of instances with the tightest bound CP-SAT will give, kept on disk.

The pool is the resolution limit of everything downstream. A success rate over
twelve instances moves in steps of one twelfth, which is coarser than the
distance being measured, and no number of seeds fixes that: every seed is scored
on the same instances, so the error they carry is common to all of them rather
than averaging away.
"""
import pickle, sys, time
import numpy as np
sys.path.insert(0, ".")
from ortools.sat.python import cp_model
from smt2020 import cut_instance, load_routes


def bound(inst, tl, objective="flowtime"):
    jobs = inst.as_tuples(); H = sum(d for j in jobs for _, d in j)
    m = cp_model.CpModel(); ends = {}
    per = {k: [] for k in range(inst.n_machines)}
    for j, ops in enumerate(jobs):
        prev = None
        for k, (mm, d) in enumerate(ops):
            s = m.NewIntVar(0, H, ""); e = m.NewIntVar(0, H, "")
            per[mm].append(m.NewIntervalVar(s, d, e, ""))
            if prev is not None:
                m.Add(s >= prev)
            prev = e; ends[j, k] = e
    for k in per:
        m.AddNoOverlap(per[k])
    last = [ends[j, len(o) - 1] for j, o in enumerate(jobs)]
    if objective == "makespan":
        sp = m.NewIntVar(0, H, "sp"); m.AddMaxEquality(sp, last); m.Minimize(sp)
    else:
        m.Minimize(sum(last))
    sv = cp_model.CpSolver(); sv.parameters.max_time_in_seconds = tl
    sv.parameters.num_search_workers = 8
    st = sv.Solve(m)
    return (int(sv.BestObjectiveBound()), int(sv.ObjectiveValue()),
            st == cp_model.OPTIMAL)


def main(n_lots=8, window=100, count=120, tl=30.0, dataset="LVHM", seed=700,
         objective="flowtime"):
    routes = load_routes(dataset)
    items, cert, proved = [], [], 0
    t0 = time.time()
    for i in range(count):
        inst = cut_instance(routes, n_lots, window,
                            np.random.default_rng(seed + i), dataset=dataset)
        lb, ub, pr = bound(inst, tl, objective)
        items.append((inst, lb)); cert.append((ub - lb) / lb); proved += pr
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{count}  {(time.time()-t0)/60:.1f} min", flush=True)
    out = f"pool_{objective}_{dataset}_{n_lots}x{window}_{count}.pkl"
    pickle.dump(items, open(out, "wb"))
    print(f"{out}: {count} instances, {proved} proved optimal, "
          f"mean certified gap {np.mean(cert):.2%}, "
          f"worst {np.max(cert):.2%}, {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    a = [int(x) if x.isdigit() else x for x in sys.argv[1:]]
    main(*a)
