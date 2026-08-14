"""Read the trajectories and say where, if anywhere, the crossing is.

The tolerance is chosen here rather than beforehand, because every instance's
distance was written down and choosing afterwards costs nothing. What it is
chosen for is stated: the tolerance that puts the crossing inside the range that
was measured, rather than before the first checkpoint or after the last. A
tolerance so loose that the process succeeds immediately and one so tight that
it never succeeds are both consistent with the same trajectories, and neither
shows anything.

The susceptibility is reported alongside because it does not depend on that
choice at all, and so says whether a transition is sharp without anyone having
to agree on what counts as success.
"""
import json
import sys

import numpy as np
from scipy import stats

from critical_scale import decide, susceptibility

THRESHOLD = 0.5
ALPHA = 0.05


def load(path="search_T.jsonl"):
    rows = [json.loads(l) for l in open(path)]
    head, ck = rows[0], rows[1:]
    return head, ck


def sigma_table(ck, eps=None, floor=None):
    """Success rate per seed at each budget, and the budgets in order.

    Two criteria. With a tolerance, an instance counts as a success when the
    read-out lands within it of the bound. With `floor` instead, an instance
    counts when the read-out is at least as good as the best of the ordinary
    dispatch rules on that same instance.

    The second has no number in it that anyone chose. A tolerance is ours, and a
    crossing that moves when we move it invites the reading that it was tuned.
    The dispatch rules are on the same instances and were not written by us, so
    they set the bar without our help.
    """
    Ts = sorted({r["instances"] for r in ck})
    seeds = sorted({r["seed"] for r in ck})
    S = np.full((len(seeds), len(Ts)), np.nan)
    for r in ck:
        i, j = seeds.index(r["seed"]), Ts.index(r["instances"])
        g = np.asarray(r["gaps"])
        S[i, j] = float((g <= eps).mean()) if floor is None \
            else float((g <= floor).mean())
    return np.asarray(Ts), seeds, S


def crossing(Ts, S):
    """The first budget whose runs are significantly above the threshold."""
    for j, T in enumerate(Ts):
        col = S[:, j]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            continue
        if decide(col, THRESHOLD, ALPHA) == "yes":
            return T, j
    return None, None


def main(path="search_T.jsonl"):
    head, ck = load(path)
    floor = np.asarray(head["floor"])
    n = len(ck[0]["gaps"])
    print(f"{len(ck)} checkpoints, {n} instances, "
          f"{len(set(r['seed'] for r in ck))} seeds", flush=True)
    print(f"dispatch rules: mean {floor.mean():.4f}\n", flush=True)

    # which tolerance puts the crossing inside what was measured
    grid = np.round(np.arange(0.02, 0.26, 0.005), 4)
    usable = []
    for e in grid:
        Ts, seeds, S = sigma_table(ck, e)
        lo, hi = np.nanmean(S[:, 0]), np.nanmean(S[:, -1])
        T_c, j = crossing(Ts, S)
        inside = T_c is not None and j > 0
        usable.append((e, lo, hi, T_c, inside))

    print(f"{'eps':>6} {'sigma at T=0':>13} {'sigma at end':>13} "
          f"{'crossing':>10}", flush=True)
    for e, lo, hi, T_c, inside in usable:
        mark = "  <- inside the measured range" if inside else ""
        c = f"{T_c:,}" if T_c is not None else "none"
        print(f"{e:>6.3f} {lo:>13.2f} {hi:>13.2f} {c:>10}{mark}", flush=True)

    good = [u for u in usable if u[4]]
    if not good:
        print("\nNo tolerance puts a crossing inside the measured range. "
              "Either the budget never reaches the threshold, or it was already "
              "past it at the first checkpoint. Widen the range rather than "
              "reading a number off this.", flush=True)
        eps = 0.10
    else:
        # the one whose crossing sits furthest from either end
        eps = max(good, key=lambda u: min(u[3], max(u[3] for u in good)))[0]
        print(f"\ntolerance {eps:.3f} puts the crossing inside the range",
              flush=True)

    Ts, seeds, S = sigma_table(ck, eps)
    print(f"\nat tolerance {eps:.3f}", flush=True)
    print(f"{'instances':>10} {'sigma':>7} {'spread':>7} {'verdict':>9} "
          f"{'per seed':>28}", flush=True)
    for j, T in enumerate(Ts):
        col = S[:, j][~np.isnan(S[:, j])]
        if len(col) == 0:
            continue
        v = decide(col, THRESHOLD, ALPHA) if len(col) > 1 else "one seed"
        per = " ".join(f"{x:.2f}" for x in col)
        print(f"{T:>10,} {col.mean():>7.2f} {col.std(ddof=1) if len(col)>1 else 0:>7.3f} "
              f"{v:>9} {per:>28}", flush=True)

    mean_sigma = np.nanmean(S, axis=0)
    pos = Ts > 0
    if pos.sum() >= 2:
        chi = susceptibility(Ts[pos], mean_sigma[pos])
        print(f"\nsusceptibility chi = {chi:.2f}  "
              f"(the paper's members span 2.5 to 182)", flush=True)

    T_c, _ = crossing(Ts, S)
    print(f"crossing at T = {T_c:,}" if T_c else "no crossing located",
          flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "search_T.jsonl")
