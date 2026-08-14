"""Where the process starts working, on both axes, and how sharply.

Panel a is the budget axis: how many labelled decisions the program has to see.
Panel b is the program axis: how many parameters it needs. Panel c is the
success region the two of them cut out, with the boundary the locator returns.
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
THRESHOLD = 0.5


def load(path="imitate.jsonl"):
    rows = [json.loads(l) for l in open(os.path.join(HERE, path))]
    return rows[0], rows[1:]


def grid(cells):
    """sigma[width][T], averaged over seeds, and the axes."""
    Ws = sorted({r["width"] for r in cells})
    Ts = sorted({r["T"] for r in cells})
    S = np.full((len(Ws), len(Ts)), np.nan)
    P = {}
    for w in Ws:
        for j, T in enumerate(Ts):
            v = [r["sigma"] for r in cells
                 if r["width"] == w and r["T"] == T]
            if v:
                S[Ws.index(w), j] = float(np.mean(v))
            for r in cells:
                if r["width"] == w:
                    P[w] = r["P"]
    return np.array(Ws), np.array(Ts), S, P


def crossing(x, y, threshold=THRESHOLD):
    """Where the curve first crosses, by log-linear interpolation."""
    for i in range(1, len(x)):
        if y[i - 1] < threshold <= y[i]:
            f = (threshold - y[i - 1]) / (y[i] - y[i - 1])
            return 10 ** (np.log10(x[i - 1]) +
                          f * (np.log10(x[i]) - np.log10(x[i - 1])))
    return None


def chi(x, y):
    d = np.abs(np.diff(y) / np.diff(np.log10(x)))
    return float(d.max() / max(y.max(), 1e-9))


def main():
    head, cells = load()
    Ws, Ts, S, P = grid(cells)
    Ps = np.array([P[w] for w in Ws])

    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.6),
                           gridspec_kw=dict(width_ratios=[1, 1, 1.1]))

    # (a) the budget axis
    a = ax[0]
    cols = plt.cm.viridis(np.linspace(0.15, 0.85, len(Ws)))
    for i, w in enumerate(Ws):
        a.plot(Ts, S[i], color=cols[i], lw=1.7, marker="o", ms=3.4,
               mec="white", mew=0.7, label=f"{P[w]:,}")
    a.axhline(THRESHOLD, color=MUTED, lw=0.9, ls=(0, (3, 2)))
    a.annotate("beats the rules on half the instances", xy=(Ts[-1], 0.51),
               fontsize=6.6, color=MUTED, va="bottom", ha="right")
    a.set_xscale("log")
    a.set_xlabel("labelled decisions seen")
    a.set_ylabel("instances where it beats the rules")
    a.set_ylim(-0.03, 1.03)
    a.legend(title="parameters", fontsize=6.3, title_fontsize=6.3,
             loc="lower right", handlelength=1.3)
    a.set_title("a   the budget axis", loc="left", pad=4)
    a.grid(True); a.set_axisbelow(True)

    # (b) the program axis
    b = ax[1]
    cols2 = plt.cm.plasma(np.linspace(0.1, 0.8, len(Ts)))
    for j, T in enumerate(Ts):
        b.plot(Ps, S[:, j], color=cols2[j], lw=1.7, marker="o", ms=3.4,
               mec="white", mew=0.7, label=f"{T:,}")
    b.axhline(THRESHOLD, color=MUTED, lw=0.9, ls=(0, (3, 2)))
    b.set_xscale("log")
    b.set_xlabel("parameters")
    b.set_ylim(-0.03, 1.03)
    b.legend(title="decisions", fontsize=6.3, title_fontsize=6.3,
             loc="upper left", handlelength=1.3)
    b.set_title("b   the program axis", loc="left", pad=4)
    b.grid(True); b.set_axisbelow(True)

    # (c) the success region
    c = ax[2]
    m = c.pcolormesh(Ts, Ps, S, cmap="RdYlBu", vmin=0, vmax=1,
                     shading="nearest")
    cs = c.contour(Ts, Ps, S, levels=[THRESHOLD], colors=[INK], linewidths=1.6)
    c.clabel(cs, fmt={THRESHOLD: "the boundary"}, fontsize=6.6)
    c.set_xscale("log"); c.set_yscale("log")
    c.set_xlabel("labelled decisions seen")
    c.set_ylabel("parameters")
    c.set_title("c   the scales that work", loc="left", pad=4)
    fig.colorbar(m, ax=c, fraction=0.046, pad=0.03,
                 label="instances beaten")

    fig.tight_layout(w_pad=1.8)
    out = os.path.join(HERE, "figures", "cutoff")
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png")
    plt.close(fig)

    print("sigma, rows are parameters and columns labelled decisions:",
          flush=True)
    print("      " + " ".join(f"{t:>8,}" for t in Ts), flush=True)
    for i, w in enumerate(Ws):
        print(f"{P[w]:>6,} " + " ".join(f"{x:8.2f}" for x in S[i]), flush=True)

    print("\ncrossings", flush=True)
    for i, w in enumerate(Ws):
        t = crossing(Ts, S[i])
        print(f"  {P[w]:>7,} parameters: "
              + (f"T_crit = {t:,.0f} decisions, chi = {chi(Ts, S[i]):.2f}"
                 if t else "never crosses"), flush=True)
    for j, T in enumerate(Ts):
        p = crossing(Ps, S[:, j])
        print(f"  {T:>7,} decisions:  "
              + (f"P_crit = {p:,.0f} parameters, chi = {chi(Ps, S[:, j]):.2f}"
                 if p else "never crosses"), flush=True)
    print("\nwrote figures/cutoff.pdf", flush=True)


if __name__ == "__main__":
    main()
