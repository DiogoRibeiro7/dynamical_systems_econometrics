# Notebook Plan

The notebooks are implemented as reproducible research entry points. Each notebook is organized around a specific research question and is expected to produce at least one figure.

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
