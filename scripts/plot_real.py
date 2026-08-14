"""Where the placement rule starts working on the thirty-eight real chains.

Panel a is the budget axis, panel b the program axis, panel c the region the
two cut out. Panel d is what this member adds that the earlier ones could not:
the boundary crossings, plotted as parameters against labelled decisions, so the
substitution between the two resources can be read off directly.
"""
import json, os
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
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#8a8985"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESHOLD, SIGMA0 = 0.5, 0.096


def grid(rows):
    Ws = sorted({r["width"] for r in rows})
    Ts = sorted({r["T"] for r in rows})
    S = np.full((len(Ws), len(Ts)), np.nan)
    P = {r["width"]: r["P"] for r in rows}
    for i, w in enumerate(Ws):
        for j, T in enumerate(Ts):
            v = [r["sigma"] for r in rows if r["width"] == w and r["T"] == T]
            if v:
                S[i, j] = float(np.mean(v))
    return np.array([P[w] for w in Ws]), np.array(Ts, float), S


def crossing(x, y, th=THRESHOLD):
    for i in range(1, len(x)):
        if y[i - 1] < th <= y[i]:
            f = (th - y[i - 1]) / (y[i] - y[i - 1])
            return 10 ** (np.log10(x[i - 1])
                          + f * (np.log10(x[i]) - np.log10(x[i - 1])))
    return None


def chi(x, y):
    return float(np.abs(np.diff(y) / np.diff(np.log10(x))).max()
                 / max(y.max(), 1e-9))


def main():
    Ps, Ts, S = grid(json.load(open(os.path.join(HERE, "results/sweep_real.json"))))
    fig, ax = plt.subplots(1, 4, figsize=(9.6, 2.55),
                           gridspec_kw=dict(width_ratios=[1, 1, 1.12, 1]))

    a = ax[0]
    c1 = plt.cm.viridis(np.linspace(0.15, 0.85, len(Ps)))
    for i, P in enumerate(Ps):
        a.plot(Ts, S[i], color=c1[i], lw=1.7, marker="o", ms=3.4, mec="white",
               mew=0.7, label=f"{P:,}")
    a.axhline(THRESHOLD, color=MUTED, lw=0.9, ls=(0, (3, 2)))
    a.axhline(SIGMA0, color=ORANGE, lw=0.9, ls=(0, (1, 2)))
    a.annotate("nothing trained", xy=(Ts[0], SIGMA0 + 0.02), fontsize=6.4,
               color=ORANGE, va="bottom")
    a.set_xscale("log")
    a.set_xlabel("labelled decisions seen")
    a.set_ylabel("chains placed no worse than the rule")
    a.set_ylim(-0.03, 1.03)
    a.legend(title="parameters", fontsize=6.3, title_fontsize=6.3,
             loc="upper left", handlelength=1.3)
    a.set_title("a   the budget axis", loc="left", pad=4)
    a.grid(True); a.set_axisbelow(True)

    b = ax[1]
    c2 = plt.cm.plasma(np.linspace(0.1, 0.8, len(Ts)))
    for j, T in enumerate(Ts):
        named = j % 2 == 0 or j == len(Ts) - 1   # a label on every other curve
        b.plot(Ps, S[:, j], color=c2[j], lw=1.7, marker="o", ms=3.4,
               mec="white", mew=0.7,
               label=f"{T:,.0f}" if named else None)
    b.axhline(THRESHOLD, color=MUTED, lw=0.9, ls=(0, (3, 2)))
    b.set_xscale("log"); b.set_xlabel("parameters"); b.set_ylim(-0.03, 1.03)
    b.legend(title="decisions", fontsize=6.3, title_fontsize=6.3,
             loc="lower right", handlelength=1.3, ncol=2, columnspacing=0.9)
    b.set_title("b   the program axis", loc="left", pad=4)
    b.grid(True); b.set_axisbelow(True)

    c = ax[2]
    m = c.pcolormesh(Ts, Ps, S, cmap="RdYlBu", vmin=0, vmax=1,
                     shading="nearest")
    cs = c.contour(Ts, Ps, S, levels=[THRESHOLD], colors=[INK], linewidths=1.6)
    c.clabel(cs, fmt={THRESHOLD: "the boundary"}, fontsize=6.6)
    c.set_xscale("log"); c.set_yscale("log")
    c.set_xlabel("labelled decisions seen"); c.set_ylabel("parameters")
    c.set_title("c   the scales that work", loc="left", pad=4)
    fig.colorbar(m, ax=c, fraction=0.046, pad=0.03, label="chains placed")

    # d: the crossings, one per program size, as a substitution curve
    d = ax[3]
    xs, ys = [], []
    for i, P in enumerate(Ps):
        t = crossing(Ts, S[i])
        if t:
            xs.append(t); ys.append(P)
    d.plot(xs, ys, color=BLUE, lw=1.7, marker="o", ms=4.5, mec="white",
           mew=0.8, zorder=3)
    w = np.array(xs) * np.array(ys)
    for g, st in ((w.min(), (0, (3, 2))), (w.max(), (0, (1, 2)))):
        t = np.array([min(xs) * 0.55, max(xs) * 1.8])
        d.plot(t, g / t, color=MUTED, lw=0.8, ls=st, zorder=1)
    d.annotate(f"$PT$ constant, {w.min():.0e} to {w.max():.0e}",
               xy=(xs[len(xs) // 2], ys[len(ys) // 2]),
               textcoords="offset points", xytext=(10, 10), fontsize=6.5,
               color=MUTED)
    d.set_xscale("log"); d.set_yscale("log")
    d.set_xlabel("labelled decisions at the crossing")
    d.set_ylabel("parameters")
    d.set_title("d   what buys what", loc="left", pad=4)
    d.grid(True); d.set_axisbelow(True)

    fig.tight_layout(w_pad=1.6)
    out = os.path.join(HERE, "figures", "cutoff_real")
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png")
    plt.close(fig)

    print("sigma, rows are parameters, columns labelled decisions:", flush=True)
    print("      " + " ".join(f"{t:>8,.0f}" for t in Ts), flush=True)
    for i, P in enumerate(Ps):
        print(f"{P:>6,} " + " ".join(f"{x:8.2f}" for x in S[i]), flush=True)
    print("\ncrossings along the budget axis", flush=True)
    for i, P in enumerate(Ps):
        t = crossing(Ts, S[i])
        print(f"  {P:>7,} parameters: "
              + (f"T_crit = {t:,.0f}, chi = {chi(Ts, S[i]):.2f}, "
                 f"P*T = {P*t:,.0f}" if t else "never crosses"), flush=True)
    print("\ncrossings along the program axis", flush=True)
    for j, T in enumerate(Ts):
        p = crossing(Ps, S[:, j])
        print(f"  {T:>7,.0f} decisions: "
              + (f"P_crit = {p:,.0f}, chi = {chi(Ps, S[:, j]):.2f}, "
                 f"P*T = {p*T:,.0f}" if p else "never crosses"), flush=True)
    print(f"\ncritical work, the smallest P*T on the boundary: {w.min():,.0f}",
          flush=True)
    n_ok = int((S >= THRESHOLD).sum())
    print(f"{n_ok} of {S.size} scales work, {n_ok/S.size:.0%} of the grid; "
          f"a scale chosen blind averages {S.mean():.2f} against "
          f"{S[S>=THRESHOLD].mean():.2f} inside the region", flush=True)
    print("wrote figures/cutoff_real.pdf", flush=True)


if __name__ == "__main__":
    main()
