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

## Repository status

This is a scaffold-first research repo. It contains:

- article draft
- mathematical notes
- repo architecture
- implementation skeleton
- tests
- coding-agent prompts
- data contracts
- notebook plan

The implementation is intentionally lightweight but complete enough for a coding agent to extend safely.

## Suggested datasets

Free sources:

- FRED / ALFRED macroeconomic series
- OECD data API
- World Bank indicators
- ECB Statistical Data Warehouse
- Yahoo Finance via yfinance for market proxies
- Stooq for financial time series

## Project structure

```text
article/                Article draft and references
configs/                YAML configs for experiments
data/                   Raw and processed data folders, ignored by git
notebooks/              Research notebooks
prompts/                Coding-agent prompts
scripts/                CLI scripts
src/dynsys_econometrics Python package
src/dynsys_econometrics/data.py
src/dynsys_econometrics/simulation.py
src/dynsys_econometrics/extremes.py
src/dynsys_econometrics/return_times.py
src/dynsys_econometrics/econometrics.py
src/dynsys_econometrics/plots.py
tests/                  Unit tests and smoke tests
```

## Installation

```bash
poetry install
```

or

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## First smoke run

```bash
python scripts/run_smoke_experiment.py
pytest
```

## Article target

Medium article, but written with academic discipline:

> From Chaos to Crisis: Using Milhazes Freitas' Dynamical-Systems Extreme Value Theory in Econometrics

