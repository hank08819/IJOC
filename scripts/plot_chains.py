"""The real supply chains, drawn as the directed acyclic graphs they are.

Three of them are drawn stage by stage, left to right from supply to customer.
A node is a stage and its area is what a unit costs to hold there; a filled node
faces a customer and carries a service time the firm has promised. The fourth
panel places all thirty-eight, because past a few hundred stages a node-link
drawing stops being readable and pretending otherwise would hide the range.

The decision the model makes is one integer per node -- the service time that
stage quotes downstream -- and the arcs are what make those integers depend on
each other. That dependence is why the placement is NP-hard on a general network
and why a chain of two thousand stages is not just a longer version of a chain
of eight.
"""
import os
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from willems import load, ELECTRONICS

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "grid.color": "#d9d9d9", "grid.linewidth": 0.6, "legend.frameon": False,
    "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
})
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#8a8985"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRAWN = ("02", "08", "12")
SHORT = {3674: "semiconductors", 3845: "electromedical apparatus",
         2043: "cereal breakfast foods"}


def layout(c):
    """x from depth, y spread within depth, ordered to keep the arcs short."""
    L, ys = c.layer, np.zeros(c.n)
    for d in range(L.max() + 1):
        ix = np.where(L == d)[0]
        if d:                       # sit each node near what feeds it
            key = [np.mean([ys[j] for j in c.pred[i]]) if c.pred[i] else 0.0
                   for i in ix]
            ix = ix[np.argsort(key)]
        k = len(ix)
        ys[ix] = (np.arange(k) - (k - 1) / 2.0) / max(k - 1, 1) * 2.0
    return L.astype(float), ys


def draw(ax, c, tag):
    xs, ys = layout(c)
    n_arc = sum(len(p) for p in c.pred)
    for i in range(c.n):
        for j in c.pred[i]:
            ax.add_patch(FancyArrowPatch(
                (xs[j], ys[j]), (xs[i], ys[i]),
                connectionstyle="arc3,rad=0.07", arrowstyle="-|>",
                mutation_scale=4.5, lw=0.45, color=MUTED, alpha=0.6,
                shrinkA=3.5, shrinkB=4.5, zorder=1))
    s = 6 + 46 * np.sqrt(c.h / max(c.h.max(), 1e-9))
    dem = np.zeros(c.n, dtype=bool)
    dem[c.demand] = True
    ax.scatter(xs[~dem], ys[~dem], s=s[~dem], c="white", edgecolors=BLUE,
               linewidths=0.8, zorder=3)
    ax.scatter(xs[dem], ys[dem], s=s[dem], c=ORANGE, edgecolors="white",
               linewidths=0.6, zorder=4)
    ax.set_title(f"{tag}   chain {c.name}", loc="left", pad=25, color=INK)
    ax.annotate(f"{SHORT.get(c.sic, c.industry.split(',')[0].lower())}\n"
                f"{c.n} stages, {n_arc} links, depth {c.layer.max()+1}",
                xy=(0.0, 1.01), xycoords="axes fraction", fontsize=6.4,
                color=MUTED, va="bottom", linespacing=1.4)
    ax.set_xlim(-0.55, xs.max() + 0.55)
    ax.set_ylim(-1.3, 1.5)
    ax.axis("off")


def census(ax, cs, tag):
    n = np.array([c.n for c in cs], float)
    a = np.array([sum(len(p) for p in c.pred) for c in cs], float)
    el = np.array([c.sic in ELECTRONICS for c in cs])
    ax.scatter(n[~el], a[~el], s=16, c="white", edgecolors=BLUE,
               linewidths=0.9, zorder=3, label="other industries")
    ax.scatter(n[el], a[el], s=20, c=BLUE, edgecolors="white",
               linewidths=0.6, zorder=4, label="chips and electronics")
    semi = np.array([c.sic == 3674 for c in cs])
    ax.scatter(n[semi], a[semi], s=34, c=ORANGE, edgecolors="white",
               linewidths=0.7, zorder=5, label="semiconductors")
    ax.legend(fontsize=6.1, loc="upper left", handlelength=1.0,
              handletextpad=0.4, borderpad=0.2, labelspacing=0.3)
    g = np.array([1, 3000.])
    ax.plot(g, g, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.annotate("one link per stage", (60, 42), fontsize=6.3, color=MUTED,
                ha="left", va="top", rotation=31)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("stages"); ax.set_ylabel("links")
    ax.set_title(f"{tag}   all thirty-eight", loc="left", pad=4, color=INK)
    ax.grid(True); ax.set_axisbelow(True)


def main():
    cs = load()
    by = {c.name: c for c in cs}
    fig, ax = plt.subplots(1, 4, figsize=(7.5, 2.35),
                           gridspec_kw=dict(width_ratios=[0.8, 1, 1.25, 1.05]))
    for a, nm, tag in zip(ax, DRAWN, "abc"):
        draw(a, by[nm], tag)
    census(ax[3], cs, "d")
    fig.text(0.35, -0.08, "left to right: supply to customer.  node area is the "
             "cost of holding a unit there;  filled nodes face a customer.",
             ha="center", fontsize=6.5, color=MUTED)
    fig.tight_layout(w_pad=1.1)
    out = os.path.join(HERE, "figures", "chains")
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png")
    plt.close(fig)
    print(f"wrote figures/chains.pdf, {len(cs)} chains, "
          f"{min(c.n for c in cs)} to {max(c.n for c in cs)} stages", flush=True)


if __name__ == "__main__":
    main()
