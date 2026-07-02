# Dynamical Systems, Rare Events, and Econometrics

This repository supports an article and reproducible Python project inspired by the work of Jorge Milhazes Freitas and collaborators on statistical properties of dynamical systems, extreme value laws, hitting/return times, rare events, extremal index, and clustering of extremes.

The goal is not to claim that economic systems are deterministic in a simplistic sense. The goal is to build a rigorous empirical framework where economic time series are treated as observables of an evolving system, and where crises, crashes, inflation shocks, volatility bursts, and stress episodes are studied as rare events with dependence.

## Research question

Can tools from extreme value theory for dynamical systems improve econometric analysis of crisis regimes and tail dependence, compared with standard iid or weakly dependent econometric assumptions?

## Main ideas

- Economic variables are modelled as observables evaluated along a trajectory.
- Rare events are exceedances above high thresholds.
- Hitting and return times measure how long the system takes to enter a crisis region again.
- The extremal index measures clustering of extremes.
- Multivariate extremes measure whether stress appears jointly across inflation, unemployment, yields, spreads, exchange rates, and equity returns.
- Simulated chaotic maps provide controlled examples before applying the method to macro-financial data.

## Current scope

The repository currently includes:

- simulation utilities for iid, AR(1), GARCH(1,1), logistic-map, Pomeau-Manneville, and coupled-map examples
- rare-event diagnostics for exceedances, block maxima, peaks-over-threshold, runs declustering, extremal-index estimation, and threshold sensitivity
- return-time and hitting-time diagnostics, including an exponential benchmark comparison
- free-source data loaders with a standard `date`, `series_id`, `value` schema
- research notebooks for simulations, recurrence, case-study workflow, multivariate tails, and article figures
- reproducible figure-generation scripts for the article
- unit tests, smoke tests, GitHub Actions CI, `ruff`, and `mypy`

The repo is designed to support both a technical Medium article and a future formal paper.

## Quick start

Create a virtual environment and install the package with dev dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the minimal comparison script:

```bash
python scripts/run_smoke_experiment.py
```

Run the configurable synthetic experiment:

```bash
python scripts/run_experiment.py --config configs/synthetic.yaml
```

Run the configurable empirical experiment on a local canonical CSV:

```bash
python scripts/run_experiment.py --config configs/empirical.yaml
```

Generate the article figure bundle under `article/figures/`:

```bash
python scripts/generate_article_figures.py
```

Run the full quality pass:

```bash
pytest -q
ruff check .
mypy
```

## Command map

This repository currently exposes script entry points rather than a package CLI:

- `python -m dynsys_econometrics`
  - purpose: package CLI with `smoke`, `synthetic`, `empirical`, and `validate-catalog` subcommands
  - external data required: depends on subcommand
  - outputs: subcommand-specific
- `python scripts/run_smoke_experiment.py`
  - purpose: compare rare-event clustering in a logistic-map observable and an AR(1) baseline
  - external data required: no
  - outputs: terminal summary only
- `python scripts/run_experiment.py --config configs/synthetic.yaml`
  - purpose: run a reproducible synthetic experiment with saved tables, figures, and JSON summary
  - external data required: no
  - outputs: `outputs/synthetic_logistic_vs_ar1/`
- `python scripts/run_experiment.py --config configs/empirical.yaml`
  - purpose: run a reproducible empirical experiment on a local canonical CSV file
  - external data required: yes, local CSV configured at `empirical.input_path`
  - outputs: `outputs/empirical_macro_stress/`
- `python scripts/generate_article_figures.py`
  - purpose: generate the article figure bundle under `article/figures/`
  - external data required: no, but it will use `data/raw/macro_stress_example.csv` if present
  - outputs: figure PNGs and a terminal summary of generated paths
- `python scripts/fetch_public_data.py --catalog data/catalog.example.yaml`
  - purpose: validate the public-data catalog used by the offline/online ingestion layer
  - external data required: no
  - outputs: terminal validation summary only

The package CLI mirrors these core workflows:

```bash
python -m dynsys_econometrics smoke
python -m dynsys_econometrics synthetic --config configs/synthetic.yaml
python -m dynsys_econometrics empirical --config configs/empirical.yaml
python -m dynsys_econometrics validate-catalog --catalog data/catalog.example.yaml
```

## Data contract

All loaders are normalized to the same long-format schema:

```text
date, series_id, value
```

This makes the diagnostics and notebooks agnostic to the original data source.

## Suggested datasets

Free sources:

- FRED / ALFRED macroeconomic series
- OECD CSV exports
- World Bank indicators
- ECB Statistical Data Warehouse
- Yahoo Finance via yfinance for market proxies

## Project structure

```text
article/                Article draft, references, and generated figures
configs/                YAML configs for experiments
data/                   Raw and processed data folders
notebooks/              Research notebooks for simulations and empirical workflows
prompts/                Coding-agent prompts
scripts/                Smoke experiment and article-figure generation
src/dynsys_econometrics Python package for simulation, diagnostics, loaders, and plots
tests/                  Unit tests and smoke tests
```

## Notebook workflow

The notebook set is described in [notebooks/README.md](C:/Users/diogo/work_code/dynamical_systems_econometrics/notebooks/README.md). Each notebook states its research question and data assumptions explicitly.

## Runbook and reproducibility

- Researcher runbook: [docs/runbook.md](C:/Users/diogo/work_code/dynamical_systems_econometrics/docs/runbook.md)
- Reproducibility manifest: [REPRODUCIBILITY.md](C:/Users/diogo/work_code/dynamical_systems_econometrics/REPRODUCIBILITY.md)
- Code audit: [docs/code_audit.md](C:/Users/diogo/work_code/dynamical_systems_econometrics/docs/code_audit.md)
- Final code review: [docs/final_code_review.md](C:/Users/diogo/work_code/dynamical_systems_econometrics/docs/final_code_review.md)

## Limitations

- The dynamical-systems objects in this repository are benchmark processes and diagnostic inspirations, not literal models of whole economies.
- Extremal-index, recurrence, and multivariate-stress outputs are descriptive. They do not identify causal mechanisms by themselves.
- Small empirical samples can make threshold-based diagnostics unstable, so threshold sensitivity should be inspected before strong interpretation.

## Citation

For academic or technical reuse, citation metadata is provided in `CITATION.cff`.

## Article target

Medium article, but written with academic discipline:

> From Chaos to Crisis: Using Milhazes Freitas' Dynamical-Systems Extreme Value Theory in Econometrics
