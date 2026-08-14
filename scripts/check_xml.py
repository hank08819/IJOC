"""Check the reader against a second encoding of the same data.

Willems distributes the networks three ways: a workbook, a database and one XML
file per chain. The reader in willems.py takes the workbook. This reads the XML
independently and compares every value it carries -- holding cost, processing
time, demand and its standard deviation, the promised service time, and every
arc -- against what the reader produced. Two encodings agreeing is not proof
that either is right, but a transcription error in the reader would show here.
"""
import glob, os, re, sys
import numpy as np
from willems import load

XML = os.path.expanduser(
    "~/Downloads/msom.1070.0176-sm-datainxml") if len(sys.argv) < 2 else sys.argv[1]
ATTR = re.compile(r'(\w+)="([^"]*)"')


def parse(path):
    t = open(path, encoding="utf-8", errors="ignore").read()
    stages, arcs = [], []
    for m in re.finditer(r"<stage\s(.*?)/>", t, re.S):
        stages.append(dict(ATTR.findall(m.group(1))))
    for m in re.finditer(r"<arc\s(.*?)/>", t, re.S):
        d = dict(ATTR.findall(m.group(1)))
        arcs.append((d["from"], d["to"]))
    return stages, arcs


def main():
    cs = {c.name: c for c in load()}
    files = sorted(glob.glob(os.path.join(XML, "*.xml")))
    if not files:
        print(f"no XML under {XML}", flush=True)
        return
    bad = 0
    checked = {k: 0 for k in
               ("stageCost", "stageTime", "stdDevDemand", "maxServiceTime",
                "arcs", "order")}
    for f in files:
        k = os.path.basename(f)[:-4]
        c = cs[k]
        stages, arcs = parse(f)
        names = [s["stageName"] for s in stages]
        if names != list(c.name_of):
            print(f"  {k}: stage order differs"); bad += 1; continue
        checked["order"] += len(names)

        def col(a, default=0.0):
            return np.array([float(s.get(a, default) or default) for s in stages])

        for a, mine, floor in (("stageCost", c.h, 0.01),
                               ("stageTime", c.tau.astype(float), 0.0)):
            want = col(a)
            if a == "stageCost":
                want = np.maximum(want, floor)
            else:
                # the reader rounds to a whole period; the check is that it
                # rounds the same value the XML carries, not that the data set
                # is integral
                want = np.rint(want)
            if not np.allclose(want, mine, rtol=1e-9, atol=1e-9):
                d = int((~np.isclose(want, mine)).sum())
                print(f"  {k}: {a} differs on {d} stages"); bad += 1
            else:
                checked[a] += len(want)

        # demand and the promised service time, on the stages that state them
        ix = {n: i for i, n in enumerate(names)}
        for s in stages:
            i = ix[s["stageName"]]
            if "stDevDemand" in s:
                if i not in c.demand:
                    print(f"  {k}: {s['stageName']} states demand but is not "
                          f"a demand stage"); bad += 1; continue
                if not np.isclose(float(s["stDevDemand"]), c.sigma_own[i]):
                    print(f"  {k}: stDevDemand differs at {s['stageName']}")
                    bad += 1
                else:
                    checked["stdDevDemand"] += 1
                if int(float(s.get("maxServiceTime", 0))) != c.quoted[i]:
                    print(f"  {k}: maxServiceTime differs at {s['stageName']}")
                    bad += 1
                else:
                    checked["maxServiceTime"] += 1

        mine_arcs = {(names[j], names[i]) for i in range(c.n) for j in c.pred[i]}
        if mine_arcs != set(arcs):
            print(f"  {k}: arcs differ, {len(mine_arcs)} against {len(arcs)}")
            bad += 1
        else:
            checked["arcs"] += len(arcs)

    print(f"{len(files)} chains checked against the XML encoding", flush=True)
    for a, n in checked.items():
        print(f"  {n:>7,} {a} values agree", flush=True)
    print(("no disagreement" if not bad else f"{bad} DISAGREEMENTS"), flush=True)


if __name__ == "__main__":
    main()
