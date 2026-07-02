# From Chaos to Crisis: Using Milhazes Freitas' Dynamical-Systems Extreme Value Theory in Econometrics

## Abstract

Econometrics usually treats economic time series as stochastic processes with dependence: autocorrelation, conditional heteroskedasticity, structural breaks, regime changes, and non-stationarity. That framework is powerful, but it often treats crises as unusually large residuals rather than as recurrent visits to dangerous regions of the system. The work of Jorge Milhazes Freitas and collaborators on extreme value laws, rare events, hitting times, return times, and clustering in dynamical systems gives us a sharper language. It allows us to ask whether economic crises behave like isolated shocks, or whether they cluster because the system repeatedly returns to fragile regions of its own phase space.

This article proposes a computational research programme: use dynamical-systems extreme value theory as a diagnostic layer for econometrics. The goal is not to claim that economies are deterministic machines. The goal is to analyse economic observables as outputs of an evolving system and study how rare events occur, recur, cluster, and propagate across variables.

## 1. The missing object in many econometric crisis models

A standard econometric model can tell us whether inflation has persistence, whether volatility is conditionally heteroskedastic, whether GDP growth has structural breaks, or whether interest-rate spreads react to debt stress. These are useful questions. But crisis analysis often needs a different question:

> How does the system enter dangerous regions, and how often does it return there?

That question is naturally dynamical. It is about recurrence. It is about the geometry of risk. It is about the difference between a single extreme observation and a cluster of extremes.

In financial markets, this distinction is obvious. A one-day volatility spike is not the same as three months of repeated stress. In macroeconomics, a one-quarter inflation surprise is not the same as a persistent inflation regime. In sovereign debt, one spread jump is not the same as a sequence of refinancing stress episodes.

## 2. Milhazes Freitas' contribution: rare events are not just large observations

Milhazes Freitas' research programme studies statistical properties of dynamical systems, with special attention to laws of rare events, extreme value laws, hitting times, return times, point processes, and extremal clustering. The key idea is that a dynamical system generates a stochastic process when we evaluate an observable along its orbit.

In symbols, we observe:

```text
X_t = phi(f^t(x))
```

Here, `f` is the evolution rule of the system, `x` is the initial state, and `phi` is the observable. In an economic context, `phi` could be inflation, unemployment, sovereign spread, exchange-rate pressure, equity volatility, or a stress index.

The crisis region is then not merely a large residual. It is a subset of the state space. A rare event occurs when the trajectory enters that subset.

## 3. Translation into econometrics

The translation is direct but must be done carefully.

| Dynamical systems | Econometrics |
|---|---|
| State space | Latent macro-financial state |
| Orbit | Historical time path |
| Observable | Measured economic variable |
| Rare set | Crisis or stress region |
| Hitting time | Time until crisis |
| Return time | Time between crises |
| Extremal index | Degree of crisis clustering |
| Multivariate extremes | Joint stress across variables |

This gives us a disciplined way to analyse crisis recurrence. We do not need to know the full state space of the economy. We can still study the statistical traces left by recurrence and clustering in observed data.

## 4. Why the extremal index matters

The extremal index is one of the most useful concepts for econometric crisis analysis. Informally, it measures whether extreme events arrive independently or in clusters.

If the extremal index is close to 1, extremes behave more like isolated shocks. If it is clearly below 1, extremes cluster. In economics, clustering is often the real damage. A single bad quarter may be absorbed. A sequence of bad quarters changes expectations, credit conditions, defaults, unemployment, and policy responses.

This is where dynamical-systems theory is valuable. In the work of Freitas and collaborators, clustering is not an afterthought. It is tied to recurrence, periodicity, and the structure of the underlying system.

## 5. Empirical project

The repo should implement three layers.

### Layer 1: controlled simulations

Start with systems where we know the data-generating process:

- iid Gaussian baseline
- AR(1) baseline
- GARCH-like baseline
- logistic map
- intermittent map
- coupled maps for multivariate stress

The goal is to show that models with similar marginal distributions can have different recurrence and clustering structures.

### Layer 2: rare-event diagnostics

For each series, compute:

- high-threshold exceedances
- hitting times
- return times
- runs declustering
- extremal index estimates
- cluster-size distributions
- point-process diagnostics
- threshold-sensitivity analysis

### Layer 3: econometric applications

Apply the same diagnostics to:

- equity volatility
- inflation shocks
- unemployment jumps
- sovereign-spread stress
- exchange-rate pressure
- joint macro-financial stress indices

The article can focus on one example first, such as inflation and sovereign-spread stress in Portugal or the euro area, and leave the multivariate extension for the second article.

## 6. What this can prove

This framework does not prove that economics is deterministic chaos. That would be too strong and easy to attack.

It can prove something more useful:

1. rare economic events are not always well described as isolated iid extremes;
2. crisis episodes may show measurable clustering;
3. the return-time structure can differ before and after policy regimes;
4. multivariate stress can reveal joint tail dependence that standard linear correlation misses;
5. governments and central banks may change not only average outcomes, but also the recurrence structure of extreme outcomes.

That last point is politically important. Policy is not only about the mean of GDP growth or inflation. It is also about the probability, recurrence, and clustering of social damage.

## 7. Article thesis

The thesis of the article is:

> Econometric crisis analysis should not only ask how large shocks are. It should ask how the system returns to crisis regions. Milhazes Freitas' work on rare events in dynamical systems gives us the mathematical language to study that recurrence.

## 8. Proposed conclusion

Economics often speaks as if crises are external accidents. But many crises are not meteorites. They are recurrences. They are the system returning to fragile regions created by leverage, inequality, energy dependence, weak investment, housing rents, fragile supply chains, or policy choices.

Dynamical-systems extreme value theory does not replace econometrics. It strengthens it. It adds a layer that standard models often miss: the structure of rare-event recurrence.

That is the value of bringing Milhazes Freitas' work into applied econometrics. It gives us a way to move from describing shocks to measuring how economies revisit danger.
