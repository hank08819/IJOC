"""What the two changes are worth, on instances whose optima are proved.

Panel a keeps the two apart. Most of the distance to the dispatch rules is
closed by restricting the choice at each step and sampling, with no network
involved at all; the network closes the rest. Reporting only the total would
credit the learning with work the restriction did.
"""
import json
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


def main():
    R = json.load(open(os.path.join(HERE, "results.json")))
    mg, sg, eps = R["mean_gap"], R["sigma"], np.array(R["eps"])
    n = R["n_instances"]

    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6),
                           gridspec_kw=dict(width_ratios=[1.25, 1, 1]))

    # (a) where the distance goes
    a = ax[0]
    keys = ["dispatch", "restricted_uniform", "restricted_trained"]
    labs = ["a dispatch rule\n(best of three)",
            "restricted choice,\nchosen at random",
            "restricted choice,\nchosen by the policy"]
    vals = [mg[k] * 100 for k in keys]
    cols = [MUTED, BLUE, ORANGE]
    y = np.arange(3)
    a.barh(y, vals, color=cols, height=0.55)
    for i, v in enumerate(vals):
        a.annotate(f"{v:.2f}%", xy=(v, i), xytext=(4, 0),
                   textcoords="offset points", va="center", fontsize=7.5,
                   color=cols[i], fontweight="bold")
    a.set_yticks(y); a.set_yticklabels(labs, fontsize=7)
    a.invert_yaxis()
    a.set_xlim(0, 4.3)
    a.set_xlabel("distance above the proved optimum (%)")

    d1 = (mg["dispatch"] - mg["restricted_uniform"]) / \
         (mg["dispatch"] - mg["restricted_trained"])
    a.annotate(f"{d1*100:.0f}% of the gain\nis the restriction",
               xy=(2.5, 0.62), fontsize=6.8, color=BLUE, ha="center")
    a.annotate(f"{(1-d1)*100:.0f}% is the policy",
               xy=(2.5, 1.62), fontsize=6.8, color=ORANGE, ha="center")
    a.set_title("a   where the distance goes", loc="left", pad=4)
    a.grid(axis="x"); a.set_axisbelow(True)

    # (b) how often it lands inside a tolerance
    b = ax[1]
    for k, c, lab in ((("all_lots_untrained"), MUTED, "every lot scored"),
                      ("dispatch", INK2, "dispatch rule"),
                      ("restricted_uniform", BLUE, "restricted, at random"),
                      ("restricted_trained", ORANGE, "restricted, policy")):
        b.plot(eps * 100, sg[k], color=c, lw=1.7, marker="o", ms=3.2,
               mec="white", mew=0.7, label=lab)
    b.set_xscale("log")
    b.set_xticks([0.5, 1, 2, 5, 10])
    b.set_xticklabels(["0.5", "1", "2", "5", "10"])
    b.set_xlabel("tolerance (%)")
    b.set_ylabel(f"instances inside it (of {n})")
    b.set_ylim(0, 1.04)
    b.legend(loc="lower right", fontsize=6.6, handlelength=1.4)
    b.set_title("b   how often it lands inside", loc="left", pad=4)
    b.grid(True); b.set_axisbelow(True)

    # (c) what the encoder costs
    c = ax[2]
    e = R["encoder"]
    c.plot(e["ops"], e["all_pairs"], color=MUTED, lw=1.7, marker="o", ms=3.2,
           mec="white", mew=0.7, label="every pair attends")
    c.plot(e["ops"], e["sparse"], color=GREEN, lw=1.7, marker="o", ms=3.2,
           mec="white", mew=0.7, label="route, station, instance")
    r = e["all_pairs"][-1] / e["sparse"][-1]
    c.annotate(f"{r:.0f}x", xy=(e["ops"][-1], np.sqrt(e["all_pairs"][-1] *
                                                      e["sparse"][-1])),
               xytext=(-4, 0), textcoords="offset points", ha="right",
               fontsize=8, color=INK, fontweight="bold")
    c.annotate("", xy=(e["ops"][-1], e["all_pairs"][-1]),
               xytext=(e["ops"][-1], e["sparse"][-1]),
               arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
    c.set_xscale("log"); c.set_yscale("log")
    c.set_xlabel("operations in the instance")
    c.set_ylabel("one encoding (s)")
    c.legend(loc="upper left", fontsize=6.6, handlelength=1.4)
    c.set_title("c   what the encoder costs", loc="left", pad=4)
    c.grid(True); c.set_axisbelow(True)

    fig.tight_layout(w_pad=1.8)
    out = os.path.join(HERE, "figures", "method")
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png")
    plt.close(fig)
    print("wrote figures/method.pdf")


if __name__ == "__main__":
    main()
