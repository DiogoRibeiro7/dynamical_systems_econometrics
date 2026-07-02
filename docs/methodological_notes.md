# Methodological Notes

## Conceptual translation

Milhazes Freitas' work studies statistical properties of dynamical systems, especially rare events, hitting times, return times, extreme value laws, point processes, and the extremal index. In an econometric setting, the translation is:

| Dynamical-systems concept | Econometric analogue |
|---|---|
| State space | Latent macro-financial state of the economy |
| Orbit | Observed time path of the economy |
| Observable | GDP growth, inflation, spreads, unemployment, volatility, returns |
| Rare set | Crisis region or stress threshold |
| Hitting time | Time until the system enters crisis |
| Return time | Time between crisis episodes |
| Extremal index | Degree of clustering in crises |
| Compound Poisson limit | Clustered crises rather than isolated shocks |
| Multivariate extremes | Joint stress across multiple economic variables |

## Why this matters

Standard econometric models often handle dependence through autocorrelation, conditional heteroskedasticity, state-space models, or regime-switching. Those are useful, but they may understate the structure of rare-event dependence. A dynamical-systems perspective asks whether extremes are isolated or whether the geometry and recurrence of the system make crises cluster.

## Empirical design

1. Simulate controlled systems:
   - iid baseline
   - AR(1) baseline
   - GARCH-like baseline
   - logistic map
   - intermittent map

2. Estimate rare-event diagnostics:
   - thresholds
   - exceedance counts
   - return-time distribution
   - runs extremal index
   - declustered exceedance process

3. Apply to economic data:
   - inflation shocks
   - unemployment jumps
   - bond-spread stress
   - equity volatility
   - exchange-rate stress

4. Compare methods:
   - iid EVT
   - AR/GARCH residual EVT
   - dynamical-systems rare-event diagnostics
   - multivariate tail dependence diagnostics

## Caution

This repo should avoid the naive claim that the economy is a deterministic chaotic system. The defensible claim is weaker and more interesting: economic observations may display recurrence, clustering, nonlinear dependence, and tail dependence patterns that are naturally analysed using tools developed for dynamical systems.
