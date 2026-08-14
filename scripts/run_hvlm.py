"""The same measurement on the other fab type, to see whether the scale travels.

LVHM is ten products in low volume; HVLM is two in high volume. They share the
106 tool groups and the route lengths, and nothing else: the best dispatch rule
lands 1.9 per cent above the optimum on one and 7.3 on the other, so the
contention they present is not the same contention.
"""
import json, pickle, sys, time
import numpy as np, torch
sys.path.insert(0, ".")
import imitate as IM
from efficiency import RULES, dispatch
from smt2020 import cut_instance, load_routes

DS = "HVLM"
LAB = f"labels_{DS}_4x600.pkl"
POOL = f"pool_makespan_{DS}_4x600_60.pkl"


def build():
    routes = load_routes(DS)
    data, t0 = [], time.time()
    for i in range(60):
        inst = cut_instance(routes, 4, 600, np.random.default_rng(6000 + i),
                            dataset=DS)
        tgt, opt, proved = IM.optimal_order(inst)
        if tgt is None:
            continue
        order, lab = IM.labels_from(inst, tgt)
        data.append(dict(inst=inst, opt=opt, proved=proved, labels=lab))
        if (i + 1) % 15 == 0:
            print(f"  labels {i+1}/60  {(time.time()-t0)/60:.1f} min", flush=True)
    pickle.dump(data, open(LAB, "wb"))
    return data


if __name__ == "__main__":
    try:
        data = pickle.load(open(LAB, "rb"))
    except FileNotFoundError:
        data = build()
    pool = [(d["inst"], d["opt"]) for d in data[:30]]
    train = data[30:]
    floor = np.array([(min(dispatch(i, r) for r in RULES) - o) / o
                      for i, o in pool])
    print(f"{DS}: {len(train)} training, {len(pool)} scored, "
          f"{sum(d['proved'] for d in data)}/{len(data)} proved, "
          f"rules {floor.mean():.1%} above the optimum\n", flush=True)
    hdr = " ".join(f"T={t}".rjust(9) for t in IM.T_GRID)
    print(f"{'width':>6} {'P':>8}  {hdr}", flush=True)
    rows, t0 = [], time.time()
    for w in IM.WIDTHS:
        cells = []
        for T in IM.T_GRID:
            s = []
            for sd in IM.SEEDS:
                net, used = IM.train_on(train, w, T, sd)
                sig, gap = IM.score(net, pool, floor, seed=sd)
                s.append(sig)
                rows.append(dict(width=w, P=sum(p.numel() for p in
                                                net.parameters()),
                                 T=used, seed=sd, sigma=sig, gap=gap))
            cells.append(f"{np.mean(s):9.2f}")
            with open("imitate_hvlm.jsonl", "w") as f:
                f.write(json.dumps(dict(dataset=DS, widths=list(IM.WIDTHS),
                                        T=list(IM.T_GRID),
                                        floor=floor.round(5).tolist())) + "\n")
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        P = [r["P"] for r in rows if r["width"] == w][0]
        print(f"{w:>6} {P:>8,}  " + " ".join(cells)
              + f"   [{(time.time()-t0)/60:.0f} min]", flush=True)
