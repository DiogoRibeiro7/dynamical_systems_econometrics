# Research Runbook

## What to run first

Start with the smoke experiment:

```bash
python scripts/run_smoke_experiment.py
```

This is the fastest way to confirm that the core simulation, extremes, and return-time modules are wired correctly.

## Command map

### `python scripts/run_smoke_experiment.py`

- purpose: minimal deterministic comparison between a chaotic observable and an AR(1) benchmark
- modules exercised:
  - `dynsys_econometrics.simulation`
  - `dynsys_econometrics.extremes`
  - `dynsys_econometrics.return_times`
- external data required: no
- output mode: terminal summary only

### `python scripts/generate_article_figures.py`

- purpose: build the article figure bundle under `article/figures/`
- modules exercised:
  - `dynsys_econometrics.simulation`
  - `dynsys_econometrics.extremes`
  - `dynsys_econometrics.return_times`
  - `dynsys_econometrics.data`
  - `dynsys_econometrics.plots`
- external data required: no
- optional input:
  - `data/raw/macro_stress_example.csv`
- outputs:
  - orbit/observable figure
  - threshold exceedance figure
  - clustered-versus-isolated comparison
  - return-time distribution
  - extremal-index threshold sensitivity
  - macro-financial crisis timeline
  - multivariate stress heatmap

## Data-source map

The repository is designed around open or freely accessible sources:

- FRED / ALFRED for macroeconomic and financial series
- OECD CSV exports
- World Bank indicator exports
- ECB Statistical Data Warehouse
- Yahoo Finance via `yfinance`

All external series should be normalized to the long schema:

```text
date, series_id, value
```

## Workflow diagram

```text
simulation or external time series
    -> observable construction
    -> threshold selection
    -> exceedance extraction / POT / block maxima
    -> declustering and extremal-index estimation
    -> return-time analysis
    -> multivariate stress diagnostics where relevant
    -> figures, notebooks, and article-facing outputs
```

## Exact workflows

### Replication-style baseline run

```bash
python scripts/run_smoke_experiment.py
```

Use this when you want a no-data sanity check for the core methodology.

### Full figure-generation run

```bash
python scripts/generate_article_figures.py
```

This produces the figure bundle even when no raw macro example file is available.

### Notebook workflow

1. Install the package with dev dependencies.
2. Open the notebooks in order from [notebooks/README.md](C:/Users/diogo/work_code/dynamical_systems_econometrics/notebooks/README.md).
3. Treat notebooks as exploratory and explanatory layers.
4. Treat `scripts/generate_article_figures.py` as the reproducible publication path for figures.

## Which workflows require external files

- `scripts/run_smoke_experiment.py`: no external files
- `scripts/generate_article_figures.py`: no required external files, optional `data/raw/macro_stress_example.csv`
- notebooks:
  - simulation notebooks: no external files
  - empirical case-study work: external series may be needed depending on the notebook path chosen by the researcher

## Open-access-only scope

This repository should stay within open or freely accessible data workflows. The codebase is designed so that empirical ingestion can be done from public macro-financial sources and synthetic examples when needed for tests, smoke checks, or examples.

## Troubleshooting

### Missing raw example file

If `data/raw/macro_stress_example.csv` is absent, the article-figure script falls back to a synthetic macro-stress series. This is expected behavior.

### Loader parsing issues

If an external CSV does not normalize cleanly, inspect:

- date column formatting
- decimal separators
- column naming
- missing-value markers

### Municipality-style code preservation

This repository does not use municipality-code workflows. If you are adapting external panel data for related research, preserve identifier strings explicitly rather than letting them coerce to integers.

### Threshold instability

If conclusions change materially with small threshold changes, use `threshold_sensitivity_analysis` before presenting any result as stable.

### Too few extremes

If return-time or extremal-index diagnostics become unstable, lower the threshold cautiously or increase sample length. The methods require enough exceedances to be informative.
