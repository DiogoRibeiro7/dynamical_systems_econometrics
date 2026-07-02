# Notebook Plan

The notebooks are implemented as reproducible research entry points. Each notebook is organized around a specific research question and is expected to produce at least one figure.

## Prompt-aligned notebook track

These notebooks match the phased implementation prompts and save outputs under `outputs/notebooks/<notebook_name>/`.

1. `00_method_smoke_test.ipynb`
   Minimal synthetic smoke workflow with exceedances, extremal index, and return times.

2. `01_synthetic_dynamics.ipynb`
   Compare logistic-map, AR(1), and GARCH benchmark behavior.

3. `02_empirical_data_audit.ipynb`
   Audit an empirical long-format panel or a synthetic fallback without making substantive claims.

4. `03_rare_events_and_return_times.ipynb`
   Apply exceedance and recurrence diagnostics to a single empirical or fallback series.

5. `04_multivariate_stress.ipynb`
   Build a joint-stress timeline and stress-state index from multivariate series.

6. `05_econometric_baselines.ipynb`
   Compare event diagnostics with baseline volatility-style summaries.

## Original notebook track

1. `01_simulated_systems_extremes.ipynb`  
   Compare iid, AR(1), GARCH, logistic map, and intermittent-map tail behavior.

2. `02_return_times_and_extremal_index.ipynb`  
   Estimate return times, survival behavior, and extremal index across simulated systems.

3. `03_macro_financial_stress_case_study.ipynb`  
   Apply the recurrence workflow to a macro-financial stress proxy with a synthetic fallback.

4. `04_multivariate_tail_dependence.ipynb`  
   Study joint exceedances and multivariate stress in a coupled system.

5. `05_article_figures.ipynb`  
   Produce article-ready summary figures from the repository workflow.

All article-targeted figures should also be reproducible from `scripts/generate_article_figures.py`, not only from notebook execution.

## Interpretation guardrail

Empirical notebooks in this repository should frame their outputs as descriptive diagnostics of recurrence, clustering, and joint stress. They should not claim deterministic economic laws, causal identification, or exact crisis-date forecasting from these methods alone.
