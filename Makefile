PY ?= python

.PHONY: all canonical horizon stability control figures verify clean check-data

all: canonical horizon stability control figures verify

check-data:
	@sha256sum -c checksums.sha256

canonical:
	$(PY) scripts/run_canonical_s105.py

horizon:
	$(PY) scripts/run_6h_all.py

stability:
	$(PY) scripts/run_stability_10seeds.py

control:
	$(PY) scripts/run_anomaly_control.py

figures:
	$(PY) scripts/make_figures.py

verify:
	$(PY) scripts/verify_raw_results.py

clean:
	find . -name '__pycache__' -type d -exec rm -rf {} +
