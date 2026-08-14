# Everything the paper reports, from the data as distributed.
# Run from this directory.  src/ holds the model, scripts/ the drivers,
# data/ the published inputs, results/ what the drivers write.
export PYTHONPATH := src

.PHONY: all check inventory fab clean

all: check inventory

## Check the reader against the XML encoding of the same data set.
check:
	python3 scripts/check_xml.py $(XML)

## The safety-stock member on the thirty-eight published networks.
inventory:
	python3 scripts/build_real.py 180
	python3 scripts/sweep_real.py
	python3 scripts/locate_real.py
	python3 scripts/plot_real.py
	python3 scripts/plot_chains.py

## The fab members.  Needs the SMT2020 routes; see README.
fab:
	python3 scripts/imitate.py
	python3 scripts/run_hvlm.py
	python3 scripts/make_results.py

clean:
	rm -f results/labels_real.pkl
