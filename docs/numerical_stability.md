# Numerical Stability

## Guardrails implemented

- finite-value checks in univariate diagnostics
- non-positive value rejection for log returns
- zero-division avoidance in extremal-index and recurrence summaries
- duplicate-key rejection in long-format contracts

## Edge cases to treat carefully

- empty frames
- one-observation inputs
- all-equal panels
- no exceedances
- all observations above threshold
- missing values in multivariate panels

## Practical guidance

When empirical diagnostics are unstable, inspect threshold sensitivity and data coverage before drawing conclusions.
