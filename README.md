# Locating the Machine Learning Boundary in Wafer Fab Scheduling and Multiechelon Inventory

Code and data for the manuscript *Locating the Machine Learning Boundary in
Wafer Fab Scheduling and Multiechelon Inventory* by H. Han, **submitted to the
[INFORMS Journal on Computing](https://pubsonline.informs.org/journal/ijoc)**.
The code is under the [MIT License](LICENSE).

The manuscript is under review. It has not been accepted and no decision has
been made, so nothing here should be read as a journal publication. This
repository exists so that a referee, or anyone else, can check any number in
the manuscript against the runs it came from.

If the paper is accepted, this repository is deposited in the
[INFORMSJoC](https://github.com/INFORMSJoC) organisation under the manuscript's
number, with the `README`, `LICENSE` and `AUTHORS` that organisation asks for,
and DOIs are assigned to the paper and to the repository at that point.

## Cite

Until the paper appears, cite the manuscript itself and this repository by its
URL. No DOI has been assigned to either.

```
@unpublished{han2026locating,
  author = {Han, Henry},
  title  = {Locating the Machine Learning Boundary in Wafer Fab Scheduling and
            Multiechelon Inventory},
  year   = {2026},
  note   = {Manuscript submitted to the INFORMS Journal on Computing.
            Code and data at https://github.com/[repository]},
}
```

## Description

The goal of this software is to measure the scale -- the number of model
parameters and the number of labelled decisions -- at which a learned policy
starts to beat the rules an industry already runs, and to locate that boundary in
a number of training runs that grows with the logarithm of the range searched
rather than with its area.

Two operations are covered. The first is scheduling wafer lots on whole routes
from SMT2020, a published model of a real semiconductor fab. The second is
placing safety stock in a multiechelon supply network under the guaranteed-service
model, on generated networks and on the thirty-eight real networks published by
Willems.

## Contents

```
src/        the model: the policies, the exact formulations, the label
            construction, and critical_scale.py, which is the locator and is
            the same file for every experiment
scripts/    the drivers that produce every number and figure in the paper
data/       the published inputs, used as distributed
results/    what the drivers write, including the logs quoted in the paper
```

The figures are not kept here. They are outputs, and every one of them is drawn
from the files in `results/` by a script in `scripts/`; the plotting scripts
create `figures/` when they run. Keeping the drawn copies as well would mean two
versions of the same picture that can disagree.

## Building

No build is required. The code is Python and needs `numpy`, `pandas`, `torch`,
`matplotlib`, `ortools` and `scipy`. Set the source directory on the path, which
is what the `Makefile` does:

```
export PYTHONPATH=src
```

## Replicating

Run everything from the root of this repository.

The reader check of Section 5.3 needs the XML distribution of the Willems data
set, which is a separate download from the journal that published it. Point the
script at the directory of XML files:

```
make check XML=/path/to/msom.1070.0176-sm-datainxml
```

The inventory results of Sections 7.6 to 7.8, which is Table 4, Figure 5 and the
run of Algorithm 1, take about ten minutes:

```
make inventory
```

The fab results of Sections 7.1 to 7.5 need the SMT2020 routes:

```
make fab
```

The exact solver is run with one search worker and a fixed random seed. That
costs seconds and is not cosmetic: the labels are the argument of the minimum and
not the minimum, so several workers racing return whichever optimal vector
finished first, two runs then agree on the cost and disagree on the service
times, and everything trained on those labels differs afterwards. With the seed
fixed, all thirty-eight networks return identical service-time vectors on repeat.

## Results

The boundary on the thirty-eight published supply networks is in
`results/sweep_real.json`, which `scripts/plot_real.py` draws. The run of the
locator is `results/locate_real.log`. The fab sweeps are `results/imitate.jsonl`
and `results/imitate_hvlm.jsonl`, and every number in the fab sections is in
`results/results.json`.

## Which script writes which result

| in the paper | produced by |
|---|---|
| Table 1, the instances | `scripts/build_real.py`, `src/willems.py` |
| Table 2, the settings | fixed in `src/` and `scripts/`; nothing tuned |
| Table 3 and Figure 3, the fab boundary | `scripts/imitate.py`, `scripts/plot_cutoff.py` |
| Table 4 and Figure 4, the published networks | `scripts/sweep_real.py`, `scripts/plot_real.py` |
| Table 5, extrapolation | `scripts/extrapolate.py` |
| Table 6 and the method comparison | `scripts/make_results.py`, `scripts/plot_method.py` |
| the run of Algorithm 1 | `scripts/locate_real.py` |
| the reader check, 100,365 values | `scripts/check_xml.py` |
| the ported L2D environment | `src/l2d_env.py`, `scripts/crosscheck_l2d.py` |
| Figures 5 to 12, the appendices | `scripts/plot_extra.py`, `scripts/plot_extra2.py`, `scripts/plot_members.py` |

Every number quoted in the paper comes from one of these and from no other
place. The exact solver runs with a single search worker and a fixed seed, so
the labels, and therefore every table trained on them, reproduce exactly.

## Support

For support, contact Henry Han at Henry_Han@Baylor.edu.
