"""Locating a critical scale without scanning a grid.

A scan over program sizes and budgets costs the product of the two axes, and
almost all of it is spent far from the boundary it exists to find. The boundary
is one-dimensional. What follows finds it by bracketing and bisection, at a cost
that grows with the logarithm of the range rather than with its size, and it
begins by testing whether there is a boundary to find at all.

That first test is the one worth having. A training process can only have a
critical scale if it fails somewhere, and a problem can be easy enough that the
read-out succeeds with an untrained program. Then no amount of scanning finds a
transition, because there is none: the process succeeds at zero resource. One
evaluation settles it, before anything is trained.

    room()      does a scale below which this fails exist at all
    bracket()   two scales, one that fails and one that succeeds
    narrow()    halve the interval between them until it is tight
    closed()    check that success does not come back undone higher up
    locate()    the four in order

Nothing here knows what is being trained. `succeeds` is any function from a
scale to a decision, and the caller supplies it.
"""
import math

import numpy as np
from scipy import stats


def room(floor, eps, threshold):
    """Whether a critical scale can exist at this problem size.

    `floor` is the distance to the reference that the read-out achieves with an
    untrained program, one number per instance. If that already meets the
    success criterion, the process succeeds having spent nothing, and no scale
    separates failure from success. The answer is then to make the problem
    harder, not to scan further.

    Returns the success rate an untrained program reaches and whether there is
    anything left for training to do.
    """
    sigma0 = float((np.asarray(floor) <= eps).mean())
    return sigma0, sigma0 < threshold


def decide(runs, threshold, alpha=0.05):
    """Success, failure, or neither, from repeated runs at one scale.

    Each run gives a success rate over the same instances. A scale counts as
    succeeding when the rates are above the threshold by more than their spread
    accounts for, as failing when they are below it by the same margin, and as
    neither when the runs disagree. Bisection on a noisy predicate walks into
    the wrong half; returning "neither" is what stops it.
    """
    r = np.asarray(runs, dtype=float)
    if len(r) < 2:
        # One run cannot separate a scale that works from a scale that got a
        # lucky seed, and letting it try is how a search reports a boundary
        # that is not there. An earlier version returned a hard verdict here,
        # and on a partial run it duly announced a crossing from a single seed.
        return "unsure"
    if np.allclose(r, r[0]):
        return "yes" if r[0] >= threshold else "no"
    t = stats.ttest_1samp(r, threshold)
    if t.pvalue >= alpha:
        return "unsure"
    return "yes" if r.mean() > threshold else "no"


def bracket(succeeds, start, grow=2.0, cap=None, limit=12):
    """A scale that fails and a scale that succeeds, with nothing between them
    but the boundary.

    Doubling from below costs a logarithm of the distance to the boundary, and
    doubling is what keeps a badly chosen starting point from mattering.
    """
    lo, verdicts = start, {}
    v = succeeds(lo); verdicts[lo] = v
    if v == "yes":
        # already past it, so walk down instead
        for _ in range(limit):
            hi, lo = lo, lo / grow
            v = succeeds(lo); verdicts[lo] = v
            if v == "no":
                return lo, hi, verdicts
        return None, lo, verdicts            # fails nowhere in range
    for _ in range(limit):
        hi = lo * grow
        if cap is not None and hi > cap:
            return lo, None, verdicts        # succeeds nowhere in range
        v = succeeds(hi); verdicts[hi] = v
        if v == "yes":
            return lo, hi, verdicts
        lo = hi
    return lo, None, verdicts


def settled(values, delta, k=3):
    """Whether the last k values have stopped moving, in the Cauchy sense.

    The test is on the spread across seeds as the search walks outwards. Far
    from the boundary the runs at one scale agree with each other and the spread
    sits at whatever floor the estimator has, so successive scales return the
    same number. Once they do, that side is flat and doubling further buys
    nothing: the boundary is not that way.

    True when every pair among the last k values differs by less than delta.

    This decides whether to stop extending one side. It is deliberately not
    allowed to decide anything else, and in particular it never reports that a
    boundary has been found. The reason is the estimator: a spread computed from
    a handful of seeds carries a relative error of tens of per cent, so a
    tolerance small enough to be safe almost never fires and one large enough to
    fire can fire on a slope. Used to stop one side, being wrong costs a few
    runs. Used to terminate, being wrong reports a scale that is not there.

    The hard stop stays what it was: one scale that clearly fails and one that
    clearly succeeds. If both sides settle without that, the honest answer is
    that no boundary was found in the range covered.
    """
    v = list(values)[-k:]
    return len(v) >= k and (max(v) - min(v)) < delta


def noise_floor(spreads, n_seeds, k=3, mult=2.0):
    """A tolerance for settled(), from how badly the spread is itself measured.

    A spread computed from n_seeds runs is an estimate with a standard error of
    about its own size divided by the square root of twice one less than the
    seed count. Three seeds put that at half the spread, so two readings that
    differ by half again are telling us nothing.

    The first version of this took the tolerance from the same values it then
    tested, which is circular: a range is always smaller than a multiple of
    itself, so it declared every side settled after three probes, including in
    a run whose boundary the search had already stepped on.
    """
    v = [x for x in list(spreads)[-k:] if x is not None]
    if len(v) < 2 or n_seeds < 2:
        return float("inf")
    rel = 1.0 / np.sqrt(2.0 * (n_seeds - 1))
    return mult * rel * float(np.mean(v))


def bi_extend(evaluate, start, grow=2.0, limit=8, log=print,
              delta=None, k=3, n_seeds=3, threshold=0.5):
    """Extend the search in both directions until the boundary is enclosed.

    Extending upwards alone assumes the starting scale is below the boundary,
    and a starting scale is a guess. This doubles outwards on both sides until
    one end clearly fails and the other clearly succeeds, so a guess that was
    wrong in either direction costs the same logarithm.

    While no bracket exists yet, the spread across seeds says which way to look.
    Runs at the same scale agree with each other away from the boundary and
    disagree near it, so the side where the spread is larger is the side the
    boundary is on. That is worth more than the mean here, because deciding
    success needs a threshold and a threshold is a choice, whereas the spread
    rises at the transition whatever threshold is later applied.

    The spread is used to steer the search and for nothing else. Membership is
    the three conditions of the definition, and no version of it mentions the
    spread; what is measured here is where to spend the next run.

    That steer is trustworthy on the budget axis and not on the program-size
    axis, and the difference is worth stating because it was found the hard way.
    Along a budget axis the program is the same at every point, so seeds disagree
    for one reason: how close the scale is to the boundary. Along a program-size
    axis the program changes, and a smaller program is less stable to train
    whether or not it is near anything. The two causes then add. Measured on fab
    instances at 800 operations, five seeds each: width 32 succeeded on 22.7 per
    cent with a spread of 0.015, width 16 on 16.5 per cent with a spread of
    0.034. The success rate says the boundary is upwards; the spread, read
    naively, says downwards, and the success rate is the one to believe.

    None of this touches the bracket, which is decided by the verdicts and not
    by the spread. What it costs is a wasted probe when the steer points the
    wrong way on the program-size axis.

    `evaluate(scale)` returns (mean success rate, spread, verdict), where the
    verdict is "yes", "no" or "unsure". Returns the bracket, the scale whose
    spread was largest, and everything seen.
    """
    seen = {}

    def look(s):
        if s not in seen:
            seen[s] = evaluate(s)
            m, sd, v = seen[s]
            log(f"  scale {s:>10.4g}   success {m:.2f}   spread {sd:.3f}   {v}")
        return seen[s]

    def bracketed():
        f = [s for s, (m, sd, v) in seen.items() if v == "no"]
        w = [s for s, (m, sd, v) in seen.items() if v == "yes"]
        if f and w and min(w) > max(f):
            return max(f), min(w)
        return None

    def rising():
        """Whether the success rate increases with the scale, so far."""
        pts = sorted((s, seen[s][0]) for s in seen)
        return len(pts) >= 2 and all(b >= a for (_, a), (_, b)
                                     in zip(pts, pts[1:]))

    lo = hi = start
    look(start)
    down, up = [seen[start][0]], [seen[start][0]]
    alive = {"down": True, "up": True}

    for _ in range(limit):
        if (b := bracketed()) is not None:
            return b[0], b[1], seen

        # Has the success rate itself stopped moving below the threshold? If it
        # has, the answer is already known and further probes buy nothing. The
        # test is on the success rate rather than on the spread because the rate
        # is the better measured of the two by an order of magnitude, and the
        # first version of this ran the test on the spread and kept going while
        # the rate had been flat at 0.22 for three doublings.
        rates = [seen[s][0] for s in sorted(seen)]
        if len(rates) >= 3 and settled(rates, 0.05, 3) \
                and max(rates) < threshold:
            log(f"  the success rate has settled at {rates[-1]:.2f}, below the "
                f"threshold; it saturates without crossing")
            return None, None, seen

        if not (alive["down"] or alive["up"]):
            log("  both sides have settled and nothing was bracketed")
            return None, None, seen

        # Where to look next. When the rate increases with the scale and no
        # scale has succeeded yet, every scale below the smallest one tried is
        # provably worse, so going down cannot inform anything. The spread was
        # used for this before and pointed the wrong way on the program-size
        # axis, where a small program is unstable to train for reasons that have
        # nothing to do with a boundary.
        if rising() and all(v == "no" for _, _, v in seen.values()):
            if alive["down"]:
                alive["down"] = False
                log("  the rate rises with the scale and nothing has succeeded;"
                    " smaller scales cannot inform, stop going down")

        if alive["down"]:
            lo = lo / grow
            down.append(look(lo)[0])
            if settled(down, 0.05, k):
                alive["down"] = False
                log(f"  the rate has settled below {lo:g}; stop going down")
        if alive["up"]:
            hi = hi * grow
            up.append(look(hi)[0])
            if settled(up, 0.05, k):
                alive["up"] = False
                log(f"  the rate has settled above {hi:g}; stop going up")

    if (b := bracketed()) is not None:
        return b[0], b[1], seen
    return None, None, seen


def narrow(succeeds, lo, hi, ratio=1.3, limit=8):
    """Halve the bracket, on a log scale, until it is tight enough.

    A scale is a size, so the midpoint that matters is the geometric one. An
    unsure verdict ends the bisection rather than moving an end, because the
    boundary could be on either side of that probe and choosing a side would be
    guessing. The interval returned is then wider than the tolerance asked for,
    which is the honest report: the seeds no longer separate the halves.
    """
    verdicts = {}
    for _ in range(limit):
        if hi / lo <= ratio:
            break
        mid = math.sqrt(lo * hi)
        v = succeeds(mid); verdicts[mid] = v
        if v == "yes":
            hi = mid
        elif v == "no":
            lo = mid
        else:
            break                            # noise wider than the interval
    return lo, hi, verdicts


def closed(succeeds, scale, factors=(2.0, 4.0)):
    """Whether success stays once it has been reached.

    A critical scale is only a scale if spending more keeps working. If a larger
    program or a longer budget fails, what was found is a lucky point rather
    than a boundary, and the shape of the success region has to be reported
    rather than a single number.
    """
    return {f: succeeds(scale * f) for f in factors}


def locate(succeeds, start, floor=None, eps=None, threshold=0.5,
           grow=2.0, ratio=1.3, cap=None, log=print):
    """The whole procedure. Returns the interval the boundary lies in.

    `succeeds` takes a scale and returns "yes", "no" or "unsure". `floor`, when
    given, is the untrained read-out's distances, and the run stops before
    training anything if there is no room for a transition.
    """
    out = {}
    if floor is not None:
        sigma0, has_room = room(floor, eps, threshold)
        out["untrained_success_rate"] = sigma0
        log(f"untrained success rate {sigma0:.2f} against a threshold of "
            f"{threshold:.2f}")
        if not has_room:
            log("The process already succeeds untrained. No scale separates "
                "failure from success at this problem size; make the problem "
                "harder rather than scanning further.")
            out["verdict"] = "no room"
            return out

    lo, hi, v1 = bracket(succeeds, start, grow=grow, cap=cap)
    out["bracket"] = (lo, hi)
    out["verdicts"] = dict(v1)
    if lo is None:
        log(f"fails nowhere down to {hi:g}")
        out["verdict"] = "no lower end"
        return out
    if hi is None:
        log(f"succeeds nowhere up to {lo:g}")
        out["verdict"] = "no upper end"
        return out
    log(f"bracketed between {lo:g} and {hi:g}")

    lo, hi, v2 = narrow(succeeds, lo, hi, ratio=ratio)
    out["verdicts"].update(v2)
    out["interval"] = (lo, hi)
    out["estimate"] = math.sqrt(lo * hi)
    log(f"boundary between {lo:g} and {hi:g}, estimate {out['estimate']:g}")

    out["above"] = closed(succeeds, hi)
    ok = all(v == "yes" for v in out["above"].values())
    out["verdict"] = "located" if ok else "not upward closed"
    log(f"above the boundary: {out['above']} -> {out['verdict']}")
    return out


def susceptibility(scales, sigma):
    """How sharply the success rate turns, as the paper defines it.

    The largest change in the success rate per decade of scale, divided by the
    rate it settles at. It has no tolerance in it, so two processes measured
    with different tolerances can still be compared.
    """
    s = np.asarray(scales, dtype=float)
    y = np.asarray(sigma, dtype=float)
    k = np.argsort(s)
    s, y = s[k], y[k]
    d = np.abs(np.diff(y) / np.diff(np.log10(s)))
    plateau = float(np.max(y))
    return float(d.max() / plateau) if plateau > 0 else float("nan")
