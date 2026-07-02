# Reproducibility Manifest

## Scope

This repository is a reproducible research codebase for studying rare events, recurrence, and extremal clustering in simulated and macro-financial time series using dynamical-systems-inspired methods.

The repository does not require paid data, private data, or hidden credentials. The default workflows run without any external downloads.

## Python and installation

- Python: `>=3.10`
- Package install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Entry points

### Tests and quality

```bash
pytest -q
ruff check .
mypy
```

### Smoke experiment

```bash
python scripts/run_smoke_experiment.py
```

Purpose:
- runs a deterministic comparison between a logistic-map observable and an AR(1) benchmark
- prints threshold, exceedance count, cluster count, extremal index estimate, and mean return time

External data:
- none

Generated outputs:
- terminal summary only

### Article figure bundle

```bash
python scripts/generate_article_figures.py
```

Purpose:
- generates the reproducible figure bundle used by the article-facing materials

External data:
- optional only
- if `data/raw/macro_stress_example.csv` exists, it is loaded through the repository loader stack
- if that file is absent, the script falls back to a clearly synthetic macro-stress series for the timeline figure

Generated outputs:
- `article/figures/simulated_orbit_and_observable.png`
- `article/figures/threshold_exceedances.png`
- `article/figures/clustered_vs_isolated_extremes.png`
- `article/figures/return_time_distribution.png`
- `article/figures/extremal_index_by_threshold.png`
- `article/figures/macro_financial_crisis_timeline.png`
- `article/figures/multivariate_stress_heatmap.png`

## Raw and generated data layout

### Expected raw-input locations

- optional macro example input: `data/raw/macro_stress_example.csv`

### Generated-output locations

- article figures: `article/figures/`
- notebook outputs: notebook-local outputs if a user saves them during exploration

## Data requirements and redistribution

- No raw file is required for the smoke experiment.
- No raw file is required for figure generation because the script provides a synthetic fallback where necessary.
- Suggested empirical sources are open-access or freely accessible:
  - FRED / ALFRED
  - OECD CSV exports
  - World Bank indicators
  - ECB Statistical Data Warehouse
  - Yahoo Finance via `yfinance`

This repository does not redistribute third-party raw datasets by default.

## Determinism and random seeds

The reproducible scripts use fixed seeds where stochastic components are involved:

- `run_smoke_experiment.py` fixes the AR(1) seed
- `generate_article_figures.py` fixes seeds for the GARCH, iid Gaussian, coupled-map, and synthetic macro-stress helper paths

Exact floating-point values and rendered figures may vary slightly across dependency versions, but the workflow is designed to be deterministic at the code level.

## Research workflow

1. Install the package and development dependencies.
2. Run the smoke experiment to confirm the core rare-event diagnostics work without external data.
3. Run the figure-generation script to produce the article-facing visual bundle.
4. Use the notebooks for exploratory or explanatory analysis.
5. Use the loaders to bring external open-access time series into the standard `date, series_id, value` schema.

## Known limitations

- The repository currently exposes scripts rather than a package CLI.
- The empirical macro figure path uses a synthetic fallback when no example raw file is present; those outputs should not be treated as substantive empirical findings.
- The notebooks are designed as research workflows and may require user judgment about data selection and threshold choices.
- Figure rendering can differ slightly across operating systems and plotting-library versions.
