# GitHub Issue Plan

## Milestone 1. Package foundations

## Issue 1: Tighten `TimeSeriesFrame` metadata validation

### Goal

Validate optional metadata columns more explicitly.

### Files likely affected

- `src/dynsys_econometrics/contracts.py`
- `tests/test_contracts.py`

### Implementation details

- enforce string-like metadata where present
- document allowed missingness behavior

### Acceptance criteria

- optional metadata validation is explicit

### Tests required

- metadata type checks

### Dependencies / blockers

- none

## Issue 2: Add repository-level config schema examples

### Goal

Keep config files aligned with code expectations.

### Files likely affected

- `configs/default.yaml`
- `configs/synthetic.yaml`
- `configs/empirical.yaml`
- `docs/runbook.md`

### Implementation details

- add comments or examples for each required section

### Acceptance criteria

- a new user can map config keys to commands

### Tests required

- config-load smoke checks

### Dependencies / blockers

- none

## Milestone 2. Synthetic systems

## Issue 3: Add direct tests for intermittent-map qualitative behavior

### Goal

Strengthen coverage for clustered recurrence benchmarks.

### Files likely affected

- `src/dynsys_econometrics/simulation.py`
- `tests/test_simulation.py`

### Implementation details

- add broad qualitative tests with deterministic inputs

### Acceptance criteria

- intermittent-map outputs stay bounded and reproducible

### Tests required

- boundedness and length checks

### Dependencies / blockers

- none

## Issue 4: Add synthetic frame helpers for coupled-map experiments

### Goal

Make multivariate synthetic workflows easier to orchestrate.

### Files likely affected

- `src/dynsys_econometrics/simulation.py`
- `src/dynsys_econometrics/experiments.py`

### Implementation details

- add long-format wrappers around coupled-map output

### Acceptance criteria

- multivariate synthetic experiments do not need ad hoc reshaping

### Tests required

- frame-schema tests

### Dependencies / blockers

- none

## Milestone 3. Rare-event diagnostics

## Issue 5: Add no-exceedance and all-exceedance table tests

### Goal

Harden extremal diagnostics around edge cases.

### Files likely affected

- `src/dynsys_econometrics/extremes.py`
- `tests/test_extremes.py`

### Implementation details

- encode hand-crafted edge-case tables

### Acceptance criteria

- outputs are documented and stable for edge cases

### Tests required

- zero-exceedance and full-exceedance cases

### Dependencies / blockers

- none

## Issue 6: Add threshold-sensitivity figure source tables

### Goal

Make article figures traceable to CSV outputs.

### Files likely affected

- `src/dynsys_econometrics/extremes.py`
- `src/dynsys_econometrics/experiments.py`

### Implementation details

- save threshold grids as explicit experiment tables

### Acceptance criteria

- article figures do not depend on hidden intermediate state

### Tests required

- experiment-table presence tests

### Dependencies / blockers

- none

## Milestone 4. Recurrence diagnostics

## Issue 7: Distinguish event-level and cluster-level recurrence in docs

### Goal

Prevent misuse of return-time outputs.

### Files likely affected

- `docs/return_times.md`
- `docs/runbook.md`

### Implementation details

- document both workflows with examples

### Acceptance criteria

- users can tell when cluster aggregation is required

### Tests required

- none

### Dependencies / blockers

- none

## Issue 8: Add irregular-calendar recurrence tests

### Goal

Ensure recurrence calculations remain clear on non-uniform dates.

### Files likely affected

- `src/dynsys_econometrics/return_times.py`
- `tests/test_recurrence.py`

### Implementation details

- create sparse datetime fixtures

### Acceptance criteria

- duration units remain readable

### Tests required

- irregular datetime tests

### Dependencies / blockers

- none

## Milestone 5. Empirical data pipeline

## Issue 9: Freeze one curated public-data example config

### Goal

Provide a concrete empirical starter path.

### Files likely affected

- `configs/empirical.yaml`
- `docs/data_sources.md`
- `REPRODUCIBILITY.md`

### Implementation details

- document one reproducible local input layout

### Acceptance criteria

- a new user can stage one empirical example locally

### Tests required

- local-csv experiment smoke test

### Dependencies / blockers

- source selection

## Issue 10: Add directory-panel fixtures for ingestion

### Goal

Test multi-file panel loading directly.

### Files likely affected

- `tests/fixtures/`
- `tests/test_data.py`

### Implementation details

- add two-series fixture CSVs

### Acceptance criteria

- directory loads produce a validated combined panel

### Tests required

- `load_panel_from_directory` tests

### Dependencies / blockers

- none

## Milestone 6. Multivariate stress

## Issue 11: Add explicit missingness warnings in joint stress outputs

### Goal

Avoid treating missing values as silent non-events.

### Files likely affected

- `src/dynsys_econometrics/multivariate.py`
- `tests/test_multivariate.py`

### Implementation details

- add a warning or report column for incomplete dates

### Acceptance criteria

- incomplete dates are visible in output tables

### Tests required

- missing-data cases

### Dependencies / blockers

- none

## Issue 12: Add weighted stress-index documentation

### Goal

Explain how custom weights alter interpretation.

### Files likely affected

- `docs/multivariate_extremes.md`
- `docs/runbook.md`

### Implementation details

- provide examples with equal and custom weights

### Acceptance criteria

- weighting is documented as descriptive, not structural

### Tests required

- none

### Dependencies / blockers

- none

## Milestone 7. Econometric baselines

## Issue 13: Add optional dependency stubs for richer baselines

### Goal

Prepare the baseline layer for optional AR/VAR/GARCH wrappers.

### Files likely affected

- `src/dynsys_econometrics/baselines.py`
- `docs/econometric_baselines.md`

### Implementation details

- add clear import-failure messages and placeholders

### Acceptance criteria

- base installation remains lightweight

### Tests required

- informative optional-dependency tests

### Dependencies / blockers

- optional package choice

## Issue 14: Save baseline comparison tables in experiments

### Goal

Make baseline-versus-event comparisons reproducible from saved outputs.

### Files likely affected

- `src/dynsys_econometrics/experiments.py`
- `src/dynsys_econometrics/econometrics.py`

### Implementation details

- align one baseline table with event outputs and persist it

### Acceptance criteria

- baseline comparison can be plotted without rerunning analysis

### Tests required

- experiment-output table tests

### Dependencies / blockers

- none

## Milestone 8. Reproducible notebooks

## Issue 15: Align notebook naming with the prompt-target structure

### Goal

Reduce confusion between the current notebook names and the roadmap names.

### Files likely affected

- `notebooks/`
- `notebooks/README.md`

### Implementation details

- rename or add wrapper notebooks with clear mapping notes

### Acceptance criteria

- notebook navigation is unambiguous

### Tests required

- none

### Dependencies / blockers

- notebook maintenance choice

## Issue 16: Add explicit limitations cells to every empirical notebook

### Goal

Enforce the do-not-overclaim guardrail in notebook prose.

### Files likely affected

- `notebooks/*.ipynb`

### Implementation details

- add a standard limitations section to each empirical notebook

### Acceptance criteria

- notebooks avoid causal or deterministic overclaiming

### Tests required

- notebook content review

### Dependencies / blockers

- none

## Milestone 9. Article figures

## Issue 17: Add deterministic conceptual pipeline figure

### Goal

Generate the conceptual method diagram from code.

### Files likely affected

- `scripts/make_article_figures.py`
- `src/dynsys_econometrics/plots.py`
- `article/figures/README.md`

### Implementation details

- render a text-box pipeline figure with matplotlib

### Acceptance criteria

- conceptual figure is generated by script with stable filename

### Tests required

- figure-save smoke test

### Dependencies / blockers

- none

## Issue 18: Add source CSV outputs for every article figure

### Goal

Ensure every figure can be traced to structured data outputs.

### Files likely affected

- `src/dynsys_econometrics/experiments.py`
- `article/figures/README.md`

### Implementation details

- persist plotting-source tables alongside image files

### Acceptance criteria

- no figure depends on unsaved intermediate state

### Tests required

- source-table existence tests

### Dependencies / blockers

- none

## Milestone 10. Testing and CI

## Issue 19: Add direct plot helper tests for file saving

### Goal

Cover temporary-path figure output behavior explicitly.

### Files likely affected

- `tests/test_plots.py`
- `src/dynsys_econometrics/plots.py`

### Implementation details

- save figures to temporary paths and assert file creation

### Acceptance criteria

- plot helper smoke coverage includes save behavior

### Tests required

- temp-file path tests

### Dependencies / blockers

- none

## Issue 20: Add CI smoke run for experiment orchestration

### Goal

Ensure the new experiment runner remains working in automation.

### Files likely affected

- `.github/workflows/ci.yml`
- `scripts/run_experiment.py`

### Implementation details

- run a small synthetic config in CI

### Acceptance criteria

- CI covers tests and one orchestration smoke path

### Tests required

- CI execution itself

### Dependencies / blockers

- runtime budget in GitHub Actions
