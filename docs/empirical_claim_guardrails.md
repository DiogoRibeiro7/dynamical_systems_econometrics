# Empirical Claim Guardrails

## Purpose

This repository translates ideas from dynamical-systems rare-event theory into descriptive diagnostics for observed economic time series. That translation is useful, but it creates a risk of overstating what the code can justify.

## Supported claims

The code can support claims such as:

- extreme events appear more isolated or more clustered in one series than another
- recurrence to a stress region is faster or slower across benchmark systems or empirical regimes
- a multivariate stress event involves simultaneous threshold exceedances across several variables
- a baseline econometric summary and a rare-event diagnostic highlight different aspects of the same series
- threshold sensitivity matters for interpretation and should be inspected explicitly

## Unsupported claims

The code should not be used to claim that:

- an observed economy is literally a deterministic dynamical map
- recurrence diagnostics establish causality
- a low extremal index identifies a specific policy mechanism
- return times forecast the exact timing of the next crisis
- a synthetic logistic or intermittent map is a realistic macroeconomic model
- short empirical samples prove stable asymptotic extreme-value laws

## Article and notebook rule

If a sentence in a notebook, README section, or article draft implies mechanism, forecasting certainty, or causal identification, it must be backed by a separate design layer that is not present in the current repository.

## Required limitation themes

Empirical materials should include some version of these ideas:

- the diagnostics are descriptive and structural rather than causal
- economic time series are noisy, policy-sensitive, revised, and often short
- dynamical-systems inspiration does not imply deterministic-law claims about economies
- results should be interpreted as evidence about recurrence, clustering, and joint stress patterns

## Practical checklist before publishing a claim

1. Is the claim traceable to a saved table, figure, or CSV output?
2. Does the language describe a pattern rather than a mechanism?
3. Has threshold sensitivity been checked?
4. Is the result based on real empirical data rather than a synthetic fallback?
5. Would the same sentence still be defensible if the sample were shorter than expected?
