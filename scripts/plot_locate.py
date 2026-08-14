"""The three steps of locating a critical scale, and what they cost.

Panel a is measured. Panel b is a drawing of the search, not of an experiment:
the curves are what a transition looks like and the numbered points are the
order the runs are spent in. Panel c is arithmetic.
"""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "grid.color": "#d9d9d9", "grid.linewidth": 0.6, "legend.frameon": False,
    "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
})
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
HERE = os.path.dirname(os.path.abspath(__file__))

# measured, on instances cut from SMT2020, against the bound CP-SAT certified
OPS = np.array([200, 800, 1800, 4840])
UNTRAINED = np.array([0.0, 14.2, 23.4, 12.8])
RULES = np.array([0.0, 4.5, 3.6, 2.8])


def main():
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.6),
                           gridspec_kw=dict(width_ratios=[1, 1.15, 0.95]))

    # (a) step 0: is anything to be found here
    a = ax[0]
    x = np.arange(len(OPS))
    a.plot(x, UNTRAINED, color=MUTED, lw=1.7, marker="o", ms=4, mec="white",
           mew=0.8, label="untrained")
    a.plot(x, RULES, color=INK2, lw=1.7, marker="o", ms=4, mec="white",
           mew=0.8, label="an ordinary rule")
    a.fill_between(x, RULES, UNTRAINED, color=GREEN, alpha=0.18, lw=0)
    k = 2
    a.annotate(f"{UNTRAINED[k]-RULES[k]:.1f} points\nfor training to win",
               xy=(k, (UNTRAINED[k] + RULES[k]) / 2), xytext=(-8, 0),
               textcoords="offset points", ha="right", va="center",
               fontsize=6.8, color=GREEN)
    a.annotate("nothing to find:\nuntrained is optimal", xy=(0, 0.6),
               xytext=(4, 10), textcoords="offset points", fontsize=6.8,
               color=ORANGE)
    a.plot([0], [0], marker="x", ms=7, mew=1.6, color=ORANGE, ls="none")
    a.set_xticks(x); a.set_xticklabels([f"{o:,}" for o in OPS])
    a.set_xlabel("operations in the instance")
    a.set_ylabel("distance above the bound (%)")
    a.legend(loc="upper left", fontsize=6.8, handlelength=1.4)
    a.set_ylim(-1.5, 28)
    a.set_title("a   step 0, measured", loc="left", pad=4)
    a.grid(True); a.set_axisbelow(True)

    # (b) steps 1 to 3: where the runs go
    b = ax[1]
    s = np.logspace(0, 3.1, 400)
    crit = 40.0
    sig = 1 / (1 + (crit / s) ** 3.0)
    spread = np.abs(np.gradient(sig, np.log10(s)))
    spread = spread / spread.max()
    at = lambda v: 1 / (1 + (crit / v) ** 3.0)

    b.plot(s, sig, color=BLUE, lw=1.9, label="success rate")
    b.plot(s, spread, color=ORANGE, lw=1.4, ls=(0, (4, 2)),
           label="spread across seeds")
    b.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (1, 2)))
    b.axvline(crit, color=GREEN, lw=1.0)
    b.annotate("the boundary", xy=(crit, 1.20), fontsize=6.8, color=GREEN,
               ha="center")

    # 1 start, 2-5 outwards on both sides, 6-7 halving, then does it stay
    search = [(12, "1"), (6, "2"), (24, "3"), (3, "4"), (96, "5"),
              (48, "6"), (34, "7")]
    for v, lab in search:
        b.plot([v], [at(v)], marker="o", ms=9, color="white", mec=INK,
               mew=1.0, zorder=5)
        b.annotate(lab, xy=(v, at(v)), fontsize=6, color=INK, ha="center",
                   va="center", zorder=6)
    for v in (96 * 2, 96 * 4):
        b.plot([v], [at(v)], marker="s", ms=6, color=GREEN, mec="white",
               mew=0.8, zorder=5)
    b.annotate("does it stay", xy=(96 * 2.8, at(96 * 2.8)), xytext=(0, -13),
               textcoords="offset points", fontsize=6.6, color=GREEN,
               ha="center")
    b.annotate("1  start\n2-5  outwards, both sides\n6,7  halve",
               xy=(1.2, 1.14), fontsize=6.6, color=INK2, va="top")
    b.set_xscale("log")
    b.set_xlabel("scale (program size, or budget)")
    b.set_ylabel("each against its own largest")
    b.set_ylim(-0.05, 1.30)
    b.set_yticks([0, 0.5, 1.0])
    b.legend(loc="center right", fontsize=6.6, handlelength=1.5,
             bbox_to_anchor=(1.0, 0.30))
    b.set_title("b   where the runs go", loc="left", pad=4)
    b.grid(True); b.set_axisbelow(True)

    # (c) what it costs
    c = ax[2]
    rng = np.logspace(0.5, 4, 200)          # how many doublings must be covered
    m = 3
    grid = m * (np.log2(rng) + 1) ** 2       # a square grid over both axes
    here = m * (2 * np.ceil(np.log2(rng)) + 4 + 2)
    c.plot(rng, grid, color=MUTED, lw=1.8, label="a grid over both axes")
    c.plot(rng, here, color=GREEN, lw=1.8, label="Algorithm 1")
    for v, lab in ((1e2, ""), (1e4, "")):
        pass
    c.annotate(f"{grid[-1]/here[-1]:.0f}x fewer runs\nat a range of 10,000",
               xy=(1e4, here[-1]), xytext=(-6, 26), textcoords="offset points",
               ha="right", fontsize=6.8, color=GREEN)
    c.set_xscale("log"); c.set_yscale("log")
    c.set_xlabel("range the boundary could lie in")
    c.set_ylabel("training runs")
    c.legend(loc="upper left", fontsize=6.8, handlelength=1.4)
    c.set_title("c   what it costs", loc="left", pad=4)
    c.grid(True); c.set_axisbelow(True)

    fig.tight_layout(w_pad=1.9)
    out = os.path.join(HERE, "figures", "locate")
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png")
    plt.close(fig)
    print("wrote figures/locate.pdf")


if __name__ == "__main__":
    main()
