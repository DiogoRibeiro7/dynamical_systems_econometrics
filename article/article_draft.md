# Recurrence Diagnostics for Econometric Stress Episodes

## Working title

**Recurrence Diagnostics for Econometric Stress Episodes: A Methods Paper with a Cluster-Adjusted Stress Functional**

## Core thesis

Standard econometric stress diagnostics usually measure the magnitude of variation, tail losses, or event frequency. They do not directly measure the temporal organization of extremes. That omission matters because two series can have similar volatility or similar exceedance counts while differing sharply in whether extreme observations arrive as isolated shocks or as clustered stress episodes.

The paper argues that recurrence should be treated as a first-class stress object. The contribution is twofold, although the two parts are tightly connected rather than separable. On the methods side, the paper integrates exceedance incidence, return times, runs-based clustering, multivariate stress activation, and volatility baselines within one reproducible workflow. On the theory side, it introduces a cluster-adjusted stress functional that scales exceedance incidence by clustering intensity and therefore distinguishes between stress processes that generate similar event rates but different temporal concentration of extremes. The claims are intentionally modest. The paper does not present a new asymptotic theory for extremal-index estimation, nor does it offer a structural macro-financial model. Its aim is narrower and more concrete: to provide a mathematically explicit and computationally reproducible bridge between diagnostics for dependent extremes and the practical language of applied stress analysis.

## Research question

How should econometric stress analysis distinguish between the severity of extreme observations and the temporal organization of their recurrence?

Sub-questions:

1. What objects should be estimated if the analyst cares about repeated stress episodes rather than isolated large observations?
2. How can those objects be integrated with standard volatility diagnostics?
3. Can clustering information be converted into a scalar stress summary that is easy to compare across benchmark processes?

## Literature review structure

### 1. Extreme values under dependence

The starting point is extreme-value theory for dependent sequences rather than iid tail analysis. Leadbetter, Lindgren, and Rootzen, together with Embrechts, Kluppelberg, and Mikosch and later expositions such as Coles, establish the central fact that temporal dependence changes the meaning of rare events. Once extremes cluster, an event count is no longer enough. The timing and grouping of exceedances become part of the statistical object that needs to be described.

### 2. Extremal index and clustering diagnostics

O'Brien, Ferro and Segers, and Suveges provide the main theory and estimation strategies for clustering intensity in exceedance sequences. The present paper does not try to improve that literature internally. Instead, it borrows a runs-based clustering perspective and asks how such diagnostics should enter an applied stress-analysis workflow without remaining an isolated technical appendix.

### 3. Recurrence and return-time analysis

The recurrence literature in stochastic and dynamical systems treats return times and hitting times as structurally meaningful quantities. That matters here because applied stress narratives often talk about repeated crises or renewed distress without translating those intuitions into estimable objects. The return-time literature provides exactly that missing conceptual discipline.

### 4. Volatility econometrics and tail-risk baselines

ARCH and GARCH models remain essential for changing second-moment conditions, while tail-risk and systemic-risk measures such as CoVaR, SRISK, and related capital-shortfall metrics quantify severity and cross-sectional dependence. The paper does not reject those tools. It argues that they answer a different question. Volatility and tail-loss measures speak to scale and exposure; recurrence diagnostics speak to how stress is organized through time once threshold-based distress is present.

### 5. Gap

There is no shortage of theory about dependent extremes, and there is no shortage of econometric tools for volatility and systemic risk. The gap is narrower and more practical than that. What is missing is a reproducible framework in which recurrence, clustering, and return-time structure are treated as explicit stress objects and analyzed alongside conventional econometric summaries rather than outside them.

## Formal framework

Let $\{X_t\}_{t\in\mathbb{Z}}$ be a strictly stationary process with upper-tail stress interpretation.

Define:

- exceedance indicator: $I_t(u)=\mathbf{1}\{X_t>u\}$
- exceedance probability: $p(u)=\mathbb{P}(X_0>u)$
- return times between exceedances: $\tau_{j,n}=t_{j+1}-t_j$

For run length $r\ge 1$, define the cluster-start indicator:

$$
C_{t,n}^{(r)}=I_t(u_n)\prod_{j=1}^{r}(1-I_{t-j}(u_n)).
$$

The finite-level clustering coefficient is:

$$
\theta_n^{(r)}=
\frac{\mathbb{E}[C_{0,n}^{(r)}]}{\mathbb{E}[I_0(u_n)]}
=
\mathbb{P}(X_{-1}\le u_n,\dots,X_{-r}\le u_n \mid X_0>u_n).
$$

For a multivariate process $\{X_{t,j}\}_{j=1}^d$, define the stress-state index:

$$
S_t=\frac{1}{d}\sum_{j=1}^{d} \mathbf{1}\{X_{t,j}>u_{j,n}\}.
$$

This index is descriptive. It records joint stress activation, not causal propagation.

## Estimators

The empirical exceedance rate is:

$$
\widehat{p}_n(u_n)=\frac{1}{n}\sum_{t=1}^{n}I_t(u_n).
$$

The runs-based clustering estimator is:

$$
\widehat{\theta}_n^{(r)}=
\frac{\sum_{t=r+1}^{n} C_{t,n}^{(r)}}{\sum_{t=1}^{n}I_t(u_n)}.
$$

The empirical return-time distribution is:

$$
\widehat{G}_n(x)=\frac{1}{N_n-1}\sum_{j=1}^{N_n-1}\mathbf{1}\{\tau_{j,n}\le x\}.
$$

Threshold choice and run-length choice are substantive. They cannot be hidden as preprocessing details because both directly affect the measured persistence of stress.

## Original theoretical contribution

### Cluster-adjusted stress functional

Define:

$$
\Lambda(u_n,r)=\frac{p(u_n)}{\theta_n^{(r)}},
\qquad
\widehat{\Lambda}_n(u_n,r)=\frac{\widehat{p}_n(u_n)}{\widehat{\theta}_n^{(r)}}.
$$

Interpretation:

The exceedance rate measures how often the process enters the stress region, whereas the clustering coefficient measures how isolated those exceedances are once they occur. Dividing the first object by the second therefore leaves the benchmark case of independence unchanged and raises the stress load precisely when extreme events arrive in tighter temporal groups. The purpose of the functional is not to replace richer theories of tail dependence, but to convert clustering information into a scalar summary that can be compared alongside incidence and volatility measures.

### Proposition set

1. **Monotone ordering by clustering**
   - for equal exceedance probability, a process with smaller clustering coefficient has larger cluster-adjusted stress.

2. **Independent benchmark**
   - if exceedances are temporally independent, then $\theta^{(r)}(u)=1$ and $\Lambda(u,r)=p(u)$.

3. **Plug-in consistency**
   - if the exceedance-rate and clustering estimators are consistent and the denominator stays away from zero, then the plug-in functional is consistent.

4. **Run-length monotonicity**
   - if $r_2>r_1$, then $\theta_n^{(r_2)}\le \theta_n^{(r_1)}$ and therefore $\Lambda(u_n,r_2)\ge \Lambda(u_n,r_1)$.
   - this formalizes why stricter declustering rules mechanically raise the cluster-adjusted stress load.

5. **Finite-sample perturbation bound**
   - if $|\widetilde{p}-p|\le \varepsilon_p$ and $|\widetilde{\theta}-\theta|\le \varepsilon_\theta<\theta$, then

$$
\left|\frac{\widetilde{p}}{\widetilde{\theta}}-\frac{p}{\theta}\right|
\le
\frac{\varepsilon_p}{\theta-\varepsilon_\theta}
+
\frac{p\,\varepsilon_\theta}{\theta(\theta-\varepsilon_\theta)}.
$$

This gives the main numerical warning: when clustering is strong and $\theta$ is small, estimation error in the denominator is amplified.

## Synthetic evidence currently supported by the repository

### Core generated summaries

- Extremal index by series:
  - clustered synthetic stress: `0.673`
  - isolated synthetic stress: `0.780`
  - logistic benchmark observable: `1.000`
- Return-time distribution:
  - `249` return times
  - mean `20.1`
  - median `14.0`
  - 90th percentile `40.2`
  - maximum `90`
- Multivariate stress timeline:
  - `120` dated observations
  - `12` joint exceedance dates
  - maximum of `2` active series
  - mean stress score `0.150`
- Volatility baseline:
  - mean rolling volatility during exceedances `0.3429`
  - mean rolling volatility outside exceedances `0.2671`
  - exceedance count `90`

### Interpretation

The synthetic evidence supports a narrow but meaningful claim. Recurrence diagnostics separate benchmark processes that volatility alone would describe too coarsely, return-time distributions reveal dispersion that no event count can summarize adequately, multivariate stress activation identifies when stress is joint rather than isolated, and volatility remains necessary without being sufficient. The point is not that recurrence replaces standard econometric diagnostics, but that it measures a dimension of stress that those diagnostics do not directly encode.

## Threshold robustness and uncertainty

Threshold-based recurrence diagnostics carry two uncertainties at once: ordinary sampling uncertainty and design uncertainty induced by the threshold and run-length choices. That is why threshold robustness cannot be treated as a stylistic appendix item. The code now reflects that point directly by pairing the threshold scan with a moving-block bootstrap interval rather than leaving the figure as a point-estimate line.

The existing threshold-sensitivity artifact already shows the issue clearly. For the benchmark observable used in the scan, the runs-based clustering estimate is `0.888` at the `0.90` quantile with `500` exceedances, `0.909` at the `0.93` quantile with `350` exceedances, and `1.000` from the `0.95` quantile onward as the exceedance count falls from `250` to `50`. The estimate therefore appears more stable at higher thresholds precisely when the effective sample is shrinking most aggressively.

That pattern gives the uncertainty section its main point. A single clustering estimate is not enough. The updated artifacts are a first implementation step because they show both the threshold path for `theta` and the threshold path for the cluster-adjusted stress functional together with block-bootstrap uncertainty. For the threshold scan, the `Lambda` intervals are `[0.111, 0.118]`, `[0.076, 0.080]`, `[0.0498, 0.0512]`, `[0.0298, 0.0306]`, and `[0.0096, 0.0102]` as the threshold moves from `q=0.90` to `q=0.99`. A more complete applied version would still report run-length sensitivity alongside these intervals, but the core uncertainty treatment is now implemented for the functional itself rather than only for the clustering coefficient.

Run-length sensitivity is the next layer of robustness because the declustering rule is itself part of the model of persistence. In the current run-length scan at threshold quantile `0.90`, the cluster-adjusted functional is `0.100` for run lengths `1` and `2`, then rises to `0.113`, `0.149`, and `0.206` for run lengths `4`, `6`, and `8`. The corresponding bootstrap intervals are `[0.0998, 0.1006]`, `[0.1002, 0.1012]`, `[0.1115, 0.1174]`, `[0.1445, 0.1591]`, and `[0.1968, 0.2213]`. The substantive point is that a longer quiet-period requirement mechanically increases the measured stress load by merging more exceedances into common episodes. That is not a flaw in the diagnostic; it is the reason the run-length choice must be reported explicitly.

### Compact robustness summary

The manuscript now includes a compact robustness table so the main numerical results are visible in one place rather than scattered across figures and running prose. The table reports the threshold-path estimates for `Lambda` at `q=0.90, 0.93, 0.95, 0.97, 0.99` together with their bootstrap intervals, and the run-length-path estimates at `r=1, 2, 4, 6, 8` under fixed threshold quantile `0.90`. The threshold side shows how the functional becomes incidence-dominated as the cutoff rises and `theta` approaches one. The run-length side shows the opposite comparative statics: longer declustering windows raise the stress load because they absorb more exceedances into common episodes.

## Minimal real-data illustration

The repo currently contains only a short didactic real-data example, not a full empirical application, but it is still enough to show how the recurrence language works outside synthetic benchmarks.

Using the example inflation series with an upper-tail threshold at the `0.80` quantile yields `5` exceedances, all belonging to a single run from August 2021 through December 2021. Using the example equity series after converting levels into upper-tail equity losses through sign-flipped monthly log returns yields `5` exceedances spread across `5` distinct clusters between March 2020 and September 2021. Under the same coarse rule, the two series overlap in only one joint stress month: September 2021.

The point of this section is not empirical generalization. It is to show that the framework can already distinguish between persistent and intermittent stress in a real-data toy setting, and that this distinction is more informative than simply saying that both series experienced stress at some point. The code now generates a dedicated empirical illustration artifact so that the paper is not relying on prose alone for that comparison.

## Paper structure

1. Introduction
2. Literature review
3. Probabilistic framework
4. Estimators and implementation
5. Cluster-adjusted stress functional and theoretical results
6. Synthetic evidence
7. Limitations and scope
8. Conclusion

## Limits of the paper

The limits of the paper are part of its proper definition rather than an afterthought. On the theory side, the manuscript does not derive a new asymptotic theory for extremal-index estimation, solve the threshold-selection problem, or provide a general finite-sample theory for recurrence diagnostics. Its theoretical contribution is narrower: it introduces a cluster-adjusted stress functional and establishes a small set of interpretable properties around it.

On the empirical side, the paper relies primarily on synthetic evidence. That is enough to show what the diagnostics measure, but not enough to justify broad claims about real economies or financial systems. A stronger empirical paper would need uncertainty quantification, threshold and run-length robustness, and a defended real-data application.

The multivariate component is narrower still. It identifies when stress is isolated and when it is joint, but it does not identify contagion, propagation, or structural transmission.

Within those limits, the paper does something specific and defensible: it formalizes recurrence diagnostics for econometric stress analysis, integrates them with baseline volatility summaries, provides a reproducible artifact pipeline, and adds a narrow theoretical object with interpretable properties.

## What would strengthen the paper further

1. a threshold-robustness section with explicit sensitivity plots for $\widehat{\theta}_n^{(r)}$ and $\widehat{\Lambda}_n(u_n,r)$;
2. uncertainty quantification for the clustering estimates;
3. one carefully defended empirical illustration on real macro-financial data;
4. stronger positioning against existing systemic-risk dashboards and early-warning models.
