"""The same policy, with the attention replaced so a whole fab day will fit.

The member in policy.py takes Kool et al. 2019 unchanged, and that is the point
of it: nothing about the schedules it produces can be blamed on a choice of
ours. It pays for that with a full attention matrix. Every operation attends to
every other, so both time and memory grow as the square of the instance. At the
two thousand operations a solver can still close, the square costs nothing. At
the hundred thousand operations in a fab day it is ten billion entries, and the
forward pass stops fitting in memory before it stops being fast.

This file changes one thing. An operation no longer attends to every other. It
attends to three places instead, and each is a place a scheduler would look:

  its own route, a window of steps before and after, because what a lot does
    next depends on where it is in its own sequence;
  its tool group, through one summary vector per group, because what matters
    about the other lots is how much work is queued at the stations this lot
    needs;
  the instance as a whole, through one more summary, because the makespan is a
    global quantity.

The group summaries attend to each other in full. There are 106 tool groups in
SMT2020 and that number is fixed by the fab, not by how many lots are running,
so a full 106 by 106 attention is a constant cost however large the instance
gets. Long-range information still travels: two lots at opposite ends of the
instance meet at the group summary of any station they share.

Everything else about the policy is left where it was. The features are the ones
encode() in policy.py already builds. The decoder commits one operation at a
time under the same feasibility mask, and its distribution is arithmetically
identical to the one in policy.py -- the projection of the keys is hoisted out
of the loop, which is a saving and not a change.

The training rule is in train.py and it is Kool's in structure, at counts this
machine can afford: an exponential baseline first, a greedy rollout baseline
after, replacement only on a significant paired t-test at five per cent, Adam at
1e-4, no decay, no entropy term, no reward shaping, logits clipped at ten. Two
departures are recorded there rather than hidden here. The counts are smaller
than Kool's by three orders of magnitude, and the warm-up ends on its own
criterion instead of on a fixed epoch count, because copying the epoch count at
this scale hands over while the greedy roll-out is still far worse than a
sample. NOTES.md carries what was measured on both.

So the pairing is the same pairing and only the reach is different. What is new
here is the attention structure, and it is claimed as new.
"""
import numpy as np
import torch
import torch.nn as nn

from policy import CLIP, FEAT, encode          # the features are unchanged

WINDOW = 8          # steps of a lot's own route seen on each side
CHUNK = 4096        # operations scored at once, to bound peak memory


def best_device():
    """Apple's GPU when there is one, otherwise the CPU.

    Only the encoder is put there. The decoder commits one operation at a time,
    so its work per step is a few hundred numbers, and on that shape the cost of
    launching a kernel is larger than the arithmetic: measured on this machine
    the encoder runs three times faster on the GPU and the decoder twenty-four
    times slower. So the encoder runs on whatever this returns and the decoder
    runs on the CPU, which is what run_on() below sets up.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_gen(seed, device):
    """A generator the sampler can use on whichever device it runs."""
    try:
        g = torch.Generator(device=device)
    except (RuntimeError, TypeError):
        g = torch.Generator()
    g.manual_seed(seed)
    return g


def _pick(p, greedy, gen):
    """One decision from the candidate distribution, on any device."""
    if greedy:
        return int(torch.argmax(p))
    if gen is not None and gen.device != p.device:
        return int(torch.multinomial(p.cpu(), 1, generator=gen))
    return int(torch.multinomial(p, 1, generator=gen))


def build_index(inst, index, window=WINDOW, device=None):
    """Who each operation attends to, as row numbers into the feature table.

    Returns the neighbour rows, a mask marking the ones that ran off the end of
    a route, and the tool group of every operation. encode() lays the rows out
    one lot at a time, so the neighbour of row r at offset o is row r+o and the
    only question is whether it is still inside the same route.
    """
    lens = np.array([len(r) for r in inst.routes])
    k = np.concatenate([np.arange(L) for L in lens])       # step within its lot
    r = np.arange(k.size)
    off = np.arange(-window, window + 1)

    kk = k[:, None] + off[None, :]
    ok = (kk >= 0) & (kk < np.repeat(lens, lens)[:, None])
    nb = np.where(ok, r[:, None] + off[None, :], r[:, None])
    grp = np.concatenate([np.asarray(rt) for rt in inst.routes])

    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    return t(nb), t(ok), t(grp)


def _group_mean(h, grp, n_groups):
    """One summary vector per tool group, the mean of its operations."""
    d = h.shape[1]
    tot = torch.zeros(n_groups, d, dtype=h.dtype, device=h.device)
    tot.index_add_(0, grp, h)
    cnt = torch.zeros(n_groups, dtype=h.dtype, device=h.device)
    cnt.index_add_(0, grp, torch.ones_like(grp, dtype=h.dtype))
    return tot / cnt.clamp(min=1.0).unsqueeze(1)


class SparseLayer(nn.Module):
    """One encoder layer: group summaries talk, then operations read."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.gattn = nn.MultiheadAttention(d_model, n_heads, batch_first=True,
                                           dropout=0.0)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.ReLU(),
                                nn.Linear(d_model * 4, d_model))

    def forward(self, h, nb, ok, grp, n_groups):
        x = self.n1(h)
        g = _group_mean(x, grp, n_groups)
        g = g + self.gattn(g.unsqueeze(0), g.unsqueeze(0), g.unsqueeze(0),
                           need_weights=False)[0].squeeze(0)
        glob = x.mean(0, keepdim=True)

        n, d = x.shape
        out = torch.empty_like(x)
        for a in range(0, n, CHUNK):
            b = min(a + CHUNK, n)
            m = b - a
            # what this chunk of operations may look at
            ctx = torch.cat([x[nb[a:b]],                       # own route
                             g[grp[a:b]].unsqueeze(1),         # own tool group
                             glob.expand(m, 1, d)], dim=1)     # the instance
            keep = torch.cat([ok[a:b],
                              torch.ones(m, 2, dtype=torch.bool,
                                         device=x.device)], dim=1)

            q = self.q(x[a:b]).view(m, self.h, self.dh)
            k = self.k(ctx).view(m, -1, self.h, self.dh)
            v = self.v(ctx).view(m, -1, self.h, self.dh)
            s = torch.einsum('mhd,mkhd->mhk', q, k) / np.sqrt(self.dh)
            s = s.masked_fill(~keep.unsqueeze(1), float('-inf'))
            out[a:b] = self.o(torch.einsum('mhk,mkhd->mhd',
                                           torch.softmax(s, -1), v)
                              .reshape(m, d))
        h = h + out
        return h + self.ff(self.n2(h))


class SparseScheduler(nn.Module):
    """Kool's policy with the encoder attention replaced."""

    def __init__(self, d_model=64, n_heads=4, n_layers=2, window=WINDOW):
        super().__init__()
        self.window = window
        self.enc_device = torch.device("cpu")
        self.dec_device = torch.device("cpu")
        self.inp = nn.Linear(FEAT, d_model)
        self.layers = nn.ModuleList(SparseLayer(d_model, n_heads)
                                    for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.ctx = nn.Linear(d_model * 2, d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)

    def run_on(self, device):
        """Put the encoder on `device` and leave the decoder on the CPU.

        Autograd carries gradients back across the move, so training is
        unaffected by where each half sits.
        """
        self.enc_device = torch.device(device)
        self.inp.to(self.enc_device)
        self.layers.to(self.enc_device)
        self.norm.to(self.enc_device)
        self.ctx.cpu(); self.q.cpu(); self.k.cpu()
        return self

    def encode_ops(self, x, index, inst):
        dev = self.enc_device
        nb, ok, grp = build_index(inst, index, self.window, device=dev)
        h = self.inp(x.to(dev))
        for L in self.layers:
            h = L(h, nb, ok, grp, inst.n_machines)
        return self.norm(h).to(self.dec_device)

    def forward(self, x, index, inst, greedy=False, gen=None):
        """Roll out one schedule. Returns the dispatch order and its log-prob."""
        h = self.encode_ops(x, index, inst)
        graph = h.mean(0)
        jobs = inst.as_tuples()
        lens = [len(o) for o in jobs]

        # The keys do not depend on the step, so project them once. This is the
        # same arithmetic as projecting them inside the loop, only cheaper.
        K = self.k(h)
        scale = np.sqrt(h.shape[1])

        # Row number of the operation each lot would run next, kept up to date
        # instead of rebuilt, and held in ascending lot order as before.
        start = np.zeros(inst.n_jobs, dtype=np.int64)
        acc = 0
        for j, L in enumerate(lens):
            start[j] = acc
            acc += L
        next_op = [0] * inst.n_jobs
        active = list(range(inst.n_jobs))
        cand = [int(start[j]) for j in range(inst.n_jobs)]

        dev = h.device
        order, logps = [], []
        last = torch.zeros_like(graph)
        for _ in range(inst.n_ops):
            rows = [cand[j] for j in active]
            sel = torch.tensor(rows, dtype=torch.long, device=dev)
            ctx = self.ctx(torch.cat([graph, last]))
            logits = CLIP * torch.tanh((K[sel] @ self.q(ctx)) / scale)
            p = torch.softmax(logits, 0)
            a = _pick(p, greedy, gen)
            logps.append(torch.log(p[a] + 1e-12))

            j = active[a]
            k = next_op[j]
            order.append((j, k))
            last = h[cand[j]]
            next_op[j] = k + 1
            if next_op[j] < lens[j]:
                cand[j] = int(start[j]) + next_op[j]
            else:
                active.pop(a)
        return order, torch.stack(logps).sum()

    # ------------------------------------------------------------------
    # The read-out. Training needs gradients and uses forward() above. Taking
    # the best of many samples does not, and at fab-day sizes the decoder is
    # nine tenths of the running time, so it is worth writing out.
    #
    # Nothing here changes the distribution. The query at each step is
    #   q(ctx([graph ; last]))
    # and both graph and the weights are fixed for the whole roll-out, so the
    # part that depends on the step is linear in `last`, which is itself a row
    # of h. Folding those constants once turns each step into a single
    # matrix-vector product against the candidate rows.
    # ------------------------------------------------------------------
    def readout(self, x, index, inst, k=1, rng=None, greedy=False):
        """Best makespan over k sampled schedules, and the best order.

        The k roll-outs share one encoding and differ only in what they have
        committed to, so they are advanced together: one step of this loop is
        one decision in every roll-out at once.
        """
        rng = np.random.default_rng() if rng is None else rng
        with torch.no_grad():
            h = self.encode_ops(x, index, inst).numpy().astype(np.float32)
            Wc, bc = self.ctx.weight.numpy(), self.ctx.bias.numpy()
            Wq, bq = self.q.weight.numpy(), self.q.bias.numpy()
            Wk, bk = self.k.weight.numpy(), self.k.bias.numpy()

        d = h.shape[1]
        Kmat = h @ Wk.T + bk
        graph = h.mean(0)
        base = Wc[:, :d] @ graph + bc                    # the part that is fixed
        q0 = Wq @ base + bq
        M = Wq @ Wc[:, d:]                               # how `last` enters
        Mh = h @ M.T                                     # one row per operation
        scale = np.sqrt(d, dtype=np.float32)

        jobs = inst.as_tuples()
        lens = np.array([len(o) for o in jobs])
        start = np.concatenate([[0], np.cumsum(lens)[:-1]])
        mach = np.concatenate([np.asarray(r) for r in inst.routes])
        dur = np.concatenate([np.asarray(t) for t in inst.times])

        J, b = inst.n_jobs, np.arange(k)
        nxt = np.zeros((k, J), dtype=np.int64)
        cand = np.tile(start, (k, 1))
        free_m = np.zeros((k, inst.n_machines))
        free_j = np.zeros((k, J))
        qv = np.tile(q0, (k, 1))
        order = np.empty((k, inst.n_ops, 2), dtype=np.int64)

        for step in range(inst.n_ops):
            live = nxt < lens                            # jobs still unfinished
            z = np.einsum('bjd,bd->bj', Kmat[cand], qv) / scale
            z = CLIP * np.tanh(z)
            z = np.where(live, z, -np.inf)
            z -= z.max(1, keepdims=True)
            p = np.exp(z); p /= p.sum(1, keepdims=True)
            if greedy:
                a = p.argmax(1)
            else:
                # Drawn against the running total's own last value rather than
                # against one, so a float32 sum that lands just under one
                # cannot leave the draw with no candidate above it.
                c = np.cumsum(p, 1)
                a = (c > rng.random((k, 1)) * c[:, -1:]).argmax(1)

            r = cand[b, a]
            order[:, step, 0] = a
            order[:, step, 1] = nxt[b, a]
            m = mach[r]
            s = np.maximum(free_m[b, m], free_j[b, a])
            free_m[b, m] = free_j[b, a] = s + dur[r]
            qv = q0 + Mh[r]
            nxt[b, a] += 1
            cand[b, a] = np.where(nxt[b, a] < lens[a], r + 1, r)

        span = free_j.max(1)
        w = int(span.argmin())
        return float(span[w]), [(int(j), int(o)) for j, o in order[w]]


class ActiveScheduler(SparseScheduler):
    """The same encoder, deciding among fewer lots at each step.

    SparseScheduler scores every lot that could go next. There are as many of
    those as there are lots in the fab, so a hundred thousand decisions against
    six hundred lots is the part of the cost that did not come down when the
    encoder did.

    Most of those lots are not really candidates. Take the operation that would
    finish earliest of all the ones that could start, and call its station the
    one under contention. Any operation on another station can be left until
    later without pushing anything back, and on that station itself, any
    operation that could not even begin before that earliest finish is not
    competing for it yet. What is left is the contention set, and it is small:
    the fab has 106 stations and the lots spread over them.

    Restricting the choice this way is Giffler and Thompson's, from 1960. What
    it generates are the active schedules, and the reason to use it rather than
    to keep the cheapest few candidates is that no optimum is lost: an optimal
    schedule is always active, so the restriction throws away schedules that
    could not have been the answer.

    This is a different policy from the one above and is not claimed otherwise.
    It chooses from a different set at every step, so the step-by-step agreement
    with policy.py that SparseScheduler was checked for does not hold and cannot.
    What holds instead is the guarantee, which the other two do not have.
    """

    def _layout(self, inst):
        lens = np.array([len(r) for r in inst.routes])
        start = np.concatenate([[0], np.cumsum(lens)[:-1]])
        mach = np.concatenate([np.asarray(r) for r in inst.routes])
        dur = np.concatenate([np.asarray(t) for t in inst.times]).astype(float)
        return lens, start, mach, dur

    def forward(self, x, index, inst, greedy=False, gen=None, stats=None):
        """Roll out one schedule. Returns the dispatch order and its log-prob."""
        h = self.encode_ops(x, index, inst)
        graph = h.mean(0)
        K = self.k(h)
        scale = np.sqrt(h.shape[1])
        lens, start, mach, dur = self._layout(inst)

        nxt = np.zeros(inst.n_jobs, dtype=np.int64)
        cand = start.copy()
        free_m = np.zeros(inst.n_machines)
        free_j = np.zeros(inst.n_jobs)
        order, logps = [], []
        last = torch.zeros_like(graph)

        for _ in range(inst.n_ops):
            live = np.flatnonzero(nxt < lens)
            rows = cand[live]
            s = np.maximum(free_m[mach[rows]], free_j[live])
            fin = s + dur[rows]
            a = int(fin.argmin())
            m_star = mach[rows[a]]
            conf = np.flatnonzero((mach[rows] == m_star) & (s < fin[a]))
            if stats is not None:
                stats.append(len(conf))

            sel = torch.tensor(rows[conf], dtype=torch.long, device=h.device)
            ctx = self.ctx(torch.cat([graph, last]))
            p = torch.softmax(CLIP * torch.tanh((K[sel] @ self.q(ctx)) / scale), 0)
            pick = _pick(p, greedy, gen)
            logps.append(torch.log(p[pick] + 1e-12))

            j = int(live[conf[pick]])
            r = int(cand[j])
            order.append((j, int(nxt[j])))
            free_m[mach[r]] = free_j[j] = max(free_m[mach[r]], free_j[j]) + dur[r]
            last = h[r]
            nxt[j] += 1
            if nxt[j] < lens[j]:
                cand[j] = r + 1
        return order, torch.stack(logps).sum()

    def readout(self, x, index, inst, k=1, rng=None, greedy=False, stats=None,
                objective="makespan"):
        """Best makespan over k sampled schedules, all advanced together."""
        rng = np.random.default_rng() if rng is None else rng
        with torch.no_grad():
            h = self.encode_ops(x, index, inst).numpy().astype(np.float32)
            Wc, bc = self.ctx.weight.numpy(), self.ctx.bias.numpy()
            Wq, bq = self.q.weight.numpy(), self.q.bias.numpy()
            Wk, bk = self.k.weight.numpy(), self.k.bias.numpy()

        d = h.shape[1]
        Kmat = h @ Wk.T + bk
        graph = h.mean(0)
        q0 = Wq @ (Wc[:, :d] @ graph + bc) + bq
        Mh = h @ (Wq @ Wc[:, d:]).T
        scale = np.sqrt(d, dtype=np.float32)
        lens, start, mach, dur = self._layout(inst)

        J, b = inst.n_jobs, np.arange(k)
        nxt = np.zeros((k, J), dtype=np.int64)
        cand = np.tile(start, (k, 1))
        free_m = np.zeros((k, inst.n_machines))
        free_j = np.zeros((k, J))
        qv = np.tile(q0, (k, 1))
        order = np.empty((k, inst.n_ops, 2), dtype=np.int64)

        for step in range(inst.n_ops):
            live = nxt < lens
            m_of = mach[cand]
            s = np.maximum(np.take_along_axis(free_m, m_of, 1), free_j)
            fin = np.where(live, s + dur[cand], np.inf)
            a0 = fin.argmin(1)
            m_star = m_of[b, a0]
            conf = live & (m_of == m_star[:, None]) & (s < fin[b, a0][:, None])

            # Gather just the contention set, padded to the widest one. The
            # positions come from a running count rather than a sort, because
            # sorting every lot to find the few that matter costs more than the
            # scoring it was meant to save.
            cnt = conf.sum(1)
            wide = int(cnt.max())
            bi, ji = np.nonzero(conf)
            pos = (np.cumsum(conf, 1) - 1)[bi, ji]
            take = np.zeros((k, wide), dtype=np.int64)
            keep = np.zeros((k, wide), dtype=bool)
            take[bi, pos] = ji
            keep[bi, pos] = True
            rows = np.take_along_axis(cand, take, 1)
            if stats is not None:
                stats.append(cnt)

            z = np.einsum('bcd,bd->bc', Kmat[rows], qv) / scale
            z = np.where(keep, CLIP * np.tanh(z), -np.inf)
            z -= z.max(1, keepdims=True)
            p = np.exp(z); p /= p.sum(1, keepdims=True)
            if greedy:
                pick = p.argmax(1)
            else:
                c = np.cumsum(p, 1)
                pick = (c > rng.random((k, 1)) * c[:, -1:]).argmax(1)

            j = take[b, pick]
            r = cand[b, j]
            order[:, step, 0] = j
            order[:, step, 1] = nxt[b, j]
            m = mach[r]
            st = np.maximum(free_m[b, m], free_j[b, j])
            free_m[b, m] = free_j[b, j] = st + dur[r]
            qv = q0 + Mh[r]
            nxt[b, j] += 1
            cand[b, j] = np.where(nxt[b, j] < lens[j], r + 1, r)

        span = free_j.max(1) if objective == "makespan" else free_j.sum(1)
        w = int(span.argmin())
        return float(span[w]), [(int(a), int(o)) for a, o in order[w]]
