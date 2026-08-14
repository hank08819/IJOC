"""Train a scheduler on instances cut from the fab, and score what comes out.

The rule is Kool et al. 2019 and is not changed here: sample a schedule, roll
out the current baseline greedily on the same instance, and push the log
probability by the difference in makespan. The baseline is a frozen copy of the
policy, replaced only when the trained model beats it on a fixed set of
instances under a paired test. Adam at 1e-4, no decay, no entropy term, no
reward shaping, logits clipped at ten.

Two things are declared rather than fixed. The program is the scheduler, and
its size P is the number of parameters, set by the width and depth passed in.
The number of gradient steps is T. A sweep over both is what the class asks for,
and the score below is what it asks to be measured: how often the read-out lands
within a tolerance of an optimum that was proved, not estimated.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from fab import flowtime_of, makespan_of, optimal_makespan
from policy import Scheduler, encode
from policy_sparse import ActiveScheduler, SparseScheduler
from smt2020 import cut_instance, load_routes

LR = 1e-4               # Kool's, unchanged
BATCH = 8               # instances per gradient step
K_READOUT = 64          # samples in the best-of-K read-out
OBJECTIVE = "makespan"  # what a schedule is scored by; see fab.completion_times
EPOCH = 100             # gradient steps between paired tests
BASELINE_ALPHA = 0.05   # Kool's, unchanged
BETA = 0.8              # Kool's exponential baseline during warm-up
PROBE = 16              # instances the paired test is run on
WARMUP_MIN_EPOCHS = 2   # the warm-up runs at least this long, condition or not
SWITCH_DRAWS = 8        # samples per instance behind the hand-over decision
SELF_DRAWS = 4          # schedules per instance whose mean is that instance's baseline

# A single number cannot stand in for instances that differ in scale, and these
# do. Measured on the 800-operation pool: schedules of one instance differ from
# each other by 615 minutes, while instances differ from each other by 1136, so
# with a scalar baseline about three quarters of the gradient is a statement
# about which instance was drawn rather than about how it was scheduled. Kool's
# roll-out baseline does not have this problem, because it is evaluated on the
# same instance. The exponential baseline is per batch, and is theirs only for
# the first epoch; running it to the end, which is what we did, discarded the
# property that made the roll-out baseline work. Averaging SELF_DRAWS schedules
# of the same instance restores it, and the between-instance term cancels.

# Kool et al. 2019 run 2500 gradient steps to an epoch on batches of 512, warm
# up against an exponential baseline for the first epoch, and replace the
# rollout baseline against 10000 held-out instances. Those counts are set by the
# compute they had. The structure is kept here exactly -- an exponential
# baseline first, a greedy rollout baseline after, replacement only on a
# significant paired t-test at five per cent, Adam at 1e-4, no decay, no entropy
# term, no reward shaping, logits clipped at ten -- and the counts are smaller.
#
# One of those counts could not simply be scaled down. Kool's warm-up is one
# epoch, which is 1.28 million instances; one epoch here is a few thousand.
# Copying the step count copies the wrong thing, because what the warm-up is for
# is to keep the greedy roll-out from becoming the baseline while it is still
# worse than a sample -- the argmax of a nearly flat distribution commits to one
# lot and schedules badly. At Kool's scale that has stopped being true long
# before the first epoch ends. Here it has not. So the warm-up ends on the
# condition rather than on the count: it runs until the greedy roll-out is at
# least as good as the average sample on the held-out instances.


def make_net(kind, d_model, n_layers):
    """The program. Its parameter count is the resource P.

    Three of them. "full" is policy.py, Kool unchanged. "sparse" keeps that
    policy exactly and replaces only the encoder attention. "active" also
    replaces the decoder's choice set with the operations actually contending
    for the station under contention, which is a different policy and is meant
    to be.
    """
    cls = {"full": Scheduler, "sparse": SparseScheduler,
           "active": ActiveScheduler}[kind]
    return cls(d_model=d_model, n_heads=4, n_layers=n_layers)


def n_params(net):
    return sum(p.numel() for p in net.parameters())


class Pool:
    """Instances with their proved optima, computed once and kept.

    An instance whose optimum CP-SAT could not prove is dropped rather than
    used with its best bound, because a gap measured against an unproved bound
    is a gap against another heuristic.
    """

    def __init__(self, routes, n_lots, window, count, seed, time_limit=60.0):
        rng = np.random.default_rng(seed)
        self.items = []
        while len(self.items) < count:
            inst = cut_instance(routes, n_lots, window, rng)
            opt, proved = optimal_makespan(inst, time_limit=time_limit)
            if proved:
                self.items.append((inst, opt))

    def __len__(self):
        return len(self.items)


def cost_of(inst, order):
    """What a schedule is worth under the objective in force."""
    return flowtime_of(inst, order) if OBJECTIVE == "flowtime" \
        else makespan_of(inst, order)


def readout(net, inst, k=K_READOUT, rng=None):
    """Best of k sampled schedules, with the schedule that achieved it."""
    x, idx = encode(inst)
    if hasattr(net, "readout"):                  # the batched path, same maths
        return net.readout(x, idx, inst, k=k, rng=rng, objective=OBJECTIVE)
    gen = torch.Generator(); gen.manual_seed(int(rng.integers(1 << 30)))
    best, best_order = None, None
    with torch.no_grad():
        for _ in range(k):
            order, _ = net(x, idx, inst, greedy=False, gen=gen)
            m = cost_of(inst, order)
            if best is None or m < best:
                best, best_order = m, order
    return best, best_order


def gaps_of(net, pool, k=K_READOUT, seed=0):
    """Distance to the proved optimum, one number per instance.

    The schedule is re-timed by makespan_of, which asserts that the order it
    was given respects each lot's own sequence. So an illegal schedule stops
    the run instead of being scored.
    """
    rng = np.random.default_rng(seed)
    out = []
    for inst, opt in pool.items:
        m, order = readout(net, inst, k=k, rng=rng)
        assert abs(cost_of(inst, order) - m) < 1e-6, "read-out disagrees"
        out.append((m - opt) / opt)
    return np.array(out)


def score(gaps, eps):
    """Fraction of instances inside the tolerance. Any eps, after the fact."""
    return float((gaps <= eps).mean())


def geometric_checkpoints(steps, per_decade=4, first=1):
    """Step numbers to look at, spaced geometrically rather than evenly.

    A budget is a size. Evenly spaced checkpoints put every one of them in the
    top half of the range and none where a transition from nothing to something
    happens, which is at the bottom. The first run of this scan was spaced evenly
    and the whole rise from 0.08 to 0.42 fell inside the gap before its first
    checkpoint.
    """
    out, x = [], float(first)
    while x <= steps:
        k = int(round(x))
        if k and (not out or k > out[-1]):
            out.append(k)
        x *= 10 ** (1.0 / per_decade)
    if steps not in out:
        out.append(steps)
    return set(out)


def train(kind, d_model, n_layers, steps, seed, routes,
          n_lots=4, window=50, log_every=0, watch=None, watch_at=None,
          self_baseline=True):
    """T gradient steps of Kool's rule. Returns the trained policy."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 10_000)
    net = make_net(kind, d_model, n_layers)
    base = make_net(kind, d_model, n_layers)
    base.load_state_dict(net.state_dict())
    for p in base.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    gen = torch.Generator(); gen.manual_seed(seed + 1)

    # instances the paired test is run on, held fixed so the test is paired
    probe = [cut_instance(routes, n_lots, window, np.random.default_rng(7 + i))
             for i in range(PROBE)]

    if watch_at is None:
        watch_at = geometric_checkpoints(steps)

    ema, warm = None, True                       # warm-up, and its baseline
    for t in range(steps):
        opt.zero_grad()
        if warm and self_baseline:
            n_inst = max(1, BATCH // SELF_DRAWS)
            for _ in range(n_inst):
                inst = cut_instance(routes, n_lots, window, rng)
                x, idx = encode(inst)
                # Several schedules of the same instance, each scored against
                # what the others reached on it. Nothing in the advantage then
                # depends on which instance came up.
                draws = [net(x, idx, inst, greedy=False, gen=gen)
                         for _ in range(SELF_DRAWS)]
                costs = [cost_of(inst, o) for o, _ in draws]
                b = float(np.mean(costs))
                for (order, logp), c in zip(draws, costs):
                    (((c - b) / (n_inst * SELF_DRAWS)) * logp).backward()
        else:
            costs = []
            for _ in range(BATCH):
                inst = cut_instance(routes, n_lots, window, rng)
                x, idx = encode(inst)
                order, logp = net(x, idx, inst, greedy=False, gen=gen)
                cost = cost_of(inst, order)
                costs.append((cost, logp))
                if not warm:
                    with torch.no_grad():
                        b_order, _ = base(x, idx, inst, greedy=True)
                    adv = cost - cost_of(inst, b_order)
                    ((adv / BATCH) * logp).backward()
            if warm:
                mc = float(np.mean([c for c, _ in costs]))
                ema = mc if ema is None else BETA * ema + (1 - BETA) * mc
                for cost, logp in costs:
                    (((cost - ema) / BATCH) * logp).backward()
        opt.step()

        if (t + 1) % EPOCH == 0:
            with torch.no_grad():
                g = [cost_of(i, net(*encode(i), i, greedy=True)[0])
                     for i in probe]
                if warm:
                    # Averaged over several draws per instance. One draw each
                    # is noisy enough that two runs of the same configuration
                    # hand over at different steps, which makes the switch a
                    # coin toss rather than a criterion.
                    s = [float(np.mean([cost_of(i, net(*encode(i), i,
                                                           greedy=False,
                                                           gen=gen)[0])
                                        for _ in range(SWITCH_DRAWS)]))
                         for i in probe]
                else:
                    b = [cost_of(i, base(*encode(i), i, greedy=True)[0])
                         for i in probe]
            if warm:
                # Hand over once the greedy roll-out is worth being a baseline,
                # and not before the floor below, so that a single lucky epoch
                # cannot end the warm-up early.
                if (t + 1) >= WARMUP_MIN_EPOCHS * EPOCH and np.mean(g) <= np.mean(s):
                    warm = False
                    base.load_state_dict(net.state_dict())
            elif np.mean(g) < np.mean(b):
                if stats.ttest_rel(g, b).pvalue / 2 < BASELINE_ALPHA:
                    base.load_state_dict(net.state_dict())
        if watch and (t + 1) in watch_at:
            watch(t + 1, net, warm)
        if log_every and (t + 1) % log_every == 0:
            print(f"    step {t+1}/{steps}", flush=True)
    return net


def sweep(out_csv, kind="sparse", widths=(16, 24, 32, 48, 64),
          layers=2, T_grid=(3200, 6400, 12800, 25600, 51200), seeds=(0, 1, 2),
          n_lots=4, window=50, pool_size=24, report_eps=0.05):
    """The scan the class asks for: success as a function of P and T.

    T is counted in instances, not in gradient steps, because that is what the
    definition counts: the work is (T/B) applications of the rule, so T/B is the
    number of steps and T is what the run has seen. Recording steps instead
    would make the axis depend on the batch size, and two runs at the same
    nominal T would not have cost the same.

    Every instance's distance to its optimum is written out, not just the
    fraction inside one tolerance, so the tolerance can be moved afterwards
    without running any of this again.
    """
    routes = load_routes("LVHM")
    pool = Pool(routes, n_lots, window, pool_size, seed=99)
    print(f"pool: {len(pool)} instances with proved optima", flush=True)
    path, jpath = Path(out_csv), Path(out_csv).with_suffix(".gaps.jsonl")
    path.write_text("kind,d_model,n_layers,P,T,batch,steps,seed,mean_gap,secs\n")
    jpath.write_text("")
    for d in widths:
        for T in T_grid:
            steps = T // BATCH
            for s in seeds:
                t0 = time.time()
                net = train(kind, d, layers, steps, s, routes,
                            n_lots=n_lots, window=window)
                g = gaps_of(net, pool, seed=s)
                dt, P = time.time() - t0, n_params(net)
                with path.open("a") as f:
                    f.write(f"{kind},{d},{layers},{P},{T},{BATCH},{steps},{s},"
                            f"{g.mean():.4f},{dt:.1f}\n")
                with jpath.open("a") as f:
                    f.write(json.dumps(dict(kind=kind, d_model=d, P=P, T=T,
                                            batch=BATCH, steps=steps, seed=s,
                                            gaps=g.round(5).tolist())) + "\n")
                print(f"  d={d:>3} P={P:>7,} T={T:>6} ({steps} steps) seed={s}"
                      f"  sigma={score(g, report_eps):.2f} gap={g.mean():.3f}"
                      f"  {dt:.0f}s", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "probe":
        # One corner of the grid, to see the success rate move before the
        # whole scan is committed to.
        routes = load_routes("LVHM")
        pool = Pool(routes, 4, 50, 12, seed=99)
        print(f"pool: {len(pool)} instances with proved optima", flush=True)
        for T in (0, 100, 400, 800):
            t0 = time.time()
            net = train("sparse", 64, 2, T, 0, routes)
            g = gaps_of(net, pool, seed=0)
            print(f"  T={T:>4}  gap mean={g.mean():.3f} min={g.min():.3f} "
                  f"sigma(0.05)={score(g, 0.05):.2f} "
                  f"sigma(0.20)={score(g, 0.20):.2f}  {time.time()-t0:.0f}s",
                  flush=True)
    else:
        sweep(sys.argv[1] if len(sys.argv) > 1 else "sweep_sparse.csv")
