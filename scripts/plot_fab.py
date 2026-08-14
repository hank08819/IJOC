"""One figure for the data this member is built on.

The point of the figure is that nothing in it was chosen by us. The routes, the
tool groups, the processing times and the re-entrance are SMT2020's, measured
from fabs. What we chose is where to cut, and panel b is that choice made
visible rather than asserted.
"""
import json, os
import numpy as np
import matplotlib
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


def main():
    routes = json.load(open(os.path.join(HERE, "DATA/route_stats.json")))
    win = sorted(json.load(open(os.path.join(HERE, "DATA/window_all.json"))),
                 key=lambda r: r["window"])

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5),
                           gridspec_kw=dict(width_ratios=[1.15, 1, 1]))

    # (a) the fab, as published
    a = ax[0]
    names = [r["name"] for r in routes]
    steps = [r["steps"] for r in routes]
    litho = [r["litho"] for r in routes]
    y = np.arange(len(routes))
    a.barh(y, steps, color=BLUE, alpha=0.25, height=0.66,
           edgecolor=BLUE, linewidth=0.7)
    a.barh(y, litho, color=BLUE, height=0.66, edgecolor="white", linewidth=0.6)
    a.set_yticks(y); a.set_yticklabels(names)
    a.invert_yaxis()
    a.set_xlabel("process steps in the route")
    a.annotate("lithography", xy=(litho[2] / 2, 2), xytext=(0, 0),
               textcoords="offset points", fontsize=6.6, color="white",
               ha="center", va="center", fontweight="bold")
    a.annotate("every step", xy=((steps[2] + litho[2]) / 2, 2), xytext=(0, 0),
               textcoords="offset points", fontsize=6.6, color=BLUE,
               ha="center", va="center")
    a.set_title("a   ten routes, as published", loc="left", pad=4)
    a.grid(axis="x"); a.set_axisbelow(True)

    # (b) what the cut costs
    b = ax[1]
    w = [r["window"] for r in win]
    re = [r["reent"] for r in win]
    b.axhspan(3.46, 5.55, color=GREEN, alpha=0.13, lw=0)
    b.annotate("the fab's own range", xy=(5.2, 4.35), fontsize=6.8, color=GREEN)
    b.plot(w, re, color=ORANGE, lw=1.8, marker="o", ms=3.4,
           mec="white", mew=0.7)
    b.axvline(242, color=INK2, lw=0.8, ls=(0, (3, 2)))
    b.annotate("shortest\nwhole route", xy=(242, 1.5), xytext=(-6, 0),
               textcoords="offset points", fontsize=6.6, color=INK2, ha="right")
    b.set_xscale("log")
    b.set_xlabel("steps kept per lot")
    b.set_ylabel("re-entrance, visits per tool group")
    b.set_ylim(0.8, 6.0)
    b.set_title("b   a short cut is not a fab", loc="left", pad=4)
    b.grid(True); b.set_axisbelow(True)

    # (c) and it stays solvable
    c = ax[2]
    # The first cut carries CP-SAT's own start-up, not the instance's
    # difficulty, so the trend is drawn from the second point on.
    ops = [r["ops"] for r in win][1:]
    sec = [max(r["secs"], 0.005) for r in win][1:]
    c.plot(ops, sec, color=BLUE, lw=1.8, marker="o", ms=3.4,
           mec="white", mew=0.7)
    c.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (1, 2)))
    c.annotate("one second", xy=(20, 1.15), fontsize=6.8, color=MUTED)
    c.annotate("every point\nproved optimal", xy=(430, 0.008), fontsize=6.8,
               color=BLUE, ha="center")
    c.set_xscale("log"); c.set_yscale("log")
    c.set_xlabel("operations in the instance")
    c.set_ylabel("time to prove the optimum (s)")
    c.set_title("c   the optimum stays computable", loc="left", pad=4)
    c.grid(True); c.set_axisbelow(True)

    fig.tight_layout(w_pad=1.7)
    out = os.path.join(HERE, "figures", "fab_data")
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png")
    plt.close(fig)
    print("wrote figures/fab_data.pdf")


if __name__ == "__main__":
    main()
