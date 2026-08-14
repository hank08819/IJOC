"""The policy, and the rule that trains it.

Both are taken from Kool et al. 2019 and used unchanged, exactly as the lattice
member takes them: an encoder over the instance, a decoder that commits one
decision at a time under a feasibility mask, a greedy rollout baseline replaced
only when the trained model beats it under a paired test, Adam at 1e-4 without
decay, no entropy term, no reward shaping, logits clipped at ten.

Nothing here is new and nothing is meant to be. What is new is the pairing:
this rule was written for delivery vehicles, and the objective it is pointed at
is a fab schedule.
"""
import numpy as np
import torch
import torch.nn as nn

CLIP = 10.0            # Kool's logit clipping
FEAT = 8               # per-operation features, see encode()


def encode(inst):
    """One row per operation. The decoder chooses among rows still available.

    The features are the ones a scheduler can see without solving anything:
    which machine, how long, how far into its job, how much work the job has
    left, and whether the machine is the bottleneck. Machine 0 is
    photolithography and every layer returns to it, which is the fab's own
    structure rather than a choice of ours.

    The last entry is the work the lot has left in absolute terms, and it is
    there because of a measurement. Most work remaining is the strongest of the
    ordinary dispatch rules and it ranks lots by exactly that quantity. Without
    the eighth feature the quantity is present only as a product of the fourth
    and fifth, which a linear layer cannot form, and a network taught to copy
    the rule's decisions reached 64.4 per cent against a 50 per cent baseline.
    """
    jobs = inst.as_tuples()
    rows, index = [], []
    for j, ops in enumerate(jobs):
        total = sum(d for _, d in ops)
        left = total
        for k, (m, d) in enumerate(ops):
            rows.append([m / max(inst.n_machines - 1, 1),
                         d / 10.0,
                         k / max(len(ops) - 1, 1),
                         left / max(total, 1),
                         total / 100.0,
                         1.0 if m == 0 else 0.0,
                         len(ops) / 20.0,
                         left / 100.0])
            index.append((j, k))
            left -= d
    return torch.tensor(rows, dtype=torch.float32), index


class Scheduler(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.inp = nn.Linear(FEAT, d_model)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                         batch_first=True, norm_first=True,
                                         dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, n_layers)
        self.ctx = nn.Linear(d_model * 2, d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)

    def forward(self, x, index, inst, greedy=False, gen=None):
        """Roll out one schedule. Returns the dispatch order and its log-prob."""
        h = self.enc(self.inp(x).unsqueeze(0)).squeeze(0)
        graph = h.mean(0)
        jobs = inst.as_tuples()
        next_op = [0] * inst.n_jobs
        avail = {}
        for r, (j, k) in enumerate(index):
            avail.setdefault(j, {})[k] = r

        order, logps = [], []
        last = torch.zeros_like(graph)
        for _ in range(inst.n_ops):
            rows = [avail[j][next_op[j]] for j in range(inst.n_jobs)
                    if next_op[j] < len(jobs[j])]
            cand = torch.tensor(rows, dtype=torch.long)
            ctx = self.ctx(torch.cat([graph, last]))
            logits = (self.k(h[cand]) @ self.q(ctx)) / np.sqrt(h.shape[1])
            logits = CLIP * torch.tanh(logits)
            p = torch.softmax(logits, 0)
            if greedy:
                a = int(torch.argmax(p))
            else:
                a = int(torch.multinomial(p, 1, generator=gen))
            logps.append(torch.log(p[a] + 1e-12))
            r = rows[a]
            j, k = index[r]
            order.append((j, k))
            next_op[j] += 1
            last = h[r]
        return order, torch.stack(logps).sum()
