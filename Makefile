# Makefile — setup, run, test, report figures.

VENV := .venv
FROM ?= 00
NJOBS ?=

ifeq ($(OS),Windows_NT)
  BOOTSTRAP ?= py -3.12
  VPY := $(VENV)/Scripts/python.exe
else
  # Python 3.12: the pinned numpy and scipy have no wheels for 3.13 or later
  BOOTSTRAP ?= python3.12
  VPY := $(VENV)/bin/python
endif
PY := $(if $(wildcard $(VPY)),$(VPY),$(BOOTSTRAP))

.PHONY: help setup run test figures

help:
	@echo "make setup            venv from requirements.lock.txt + the 'electricity-grid' Jupyter kernel"
	@echo "make test             unit tests (leakage, data contract, features, models, calibration)"
	@echo "make run              run notebooks 00..08 in order (about 4 h; notebook 04 alone about 2 h)"
	@echo "make run FROM=05      run from notebook 05 onward"
	@echo "make run NJOBS=4      fewer worker processes, if a run stalls on Windows"
	@echo "make figures          redraw the report figures from data/processed/results.json"

setup:
	$(BOOTSTRAP) -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 'Python 3.12 is required')"
	$(BOOTSTRAP) -m venv $(VENV)
	$(VPY) -m pip install --upgrade pip
	$(VPY) -m pip install -r requirements.lock.txt
	$(VPY) -m ipykernel install --user --name electricity-grid --display-name "Python 3 (electricity-grid)"

run:
	$(PY) run_notebooks.py --from $(FROM) $(if $(NJOBS),--n-jobs $(NJOBS))

test:
	$(PY) -m pytest tests/ -q

figures:
	$(PY) report_figures/overview.py
	$(PY) report_figures/generation_mix_pacf.py
	$(PY) report_figures/leakage_audit.py
	$(PY) report_figures/feature_selection.py
	$(PY) report_figures/clustering_regimes.py
	$(PY) report_figures/regime_validation.py
	$(PY) report_figures/forecast_by_horizon.py
	$(PY) report_figures/ablation_learning_curve.py
	$(PY) report_figures/importance_measures.py
	$(PY) report_figures/conformal_coverage.py
	$(PY) report_figures/horizon_weather.py
	$(PY) report_figures/causal_designs.py
