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

For estimation, the threshold sequence should be intermediate rather than bounded-count: write $k_n = n\overline{F}(u_n)$ and assume $k_n \to \infty$ with $k_n/n \to 0$. This keeps exceedances rare but still numerous enough for consistency statements. The bounded-count regime $n\overline{F}(u_n)\to\tau\in(0,\infty)$ belongs to point-process limit theory, not to the plug-in estimation claim used here.

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
\Lambda_n(u_n,r)=\frac{p(u_n)}{\theta_n^{(r)}},
\qquad
\widehat{\Lambda}_n(u_n,r)=\frac{\widehat{p}_n(u_n)}{\widehat{\theta}_n^{(r)}}.
$$

Interpretation:

The exceedance rate measures how often the process enters the stress region, whereas the clustering coefficient measures how isolated those exceedances are once they occur. Dividing the first object by the second therefore rescales incidence by the inverse isolation probability and raises the stress load when extreme events arrive in tighter temporal groups. At a finite threshold, the honest zero-clustering reference is not `p(u)` itself but the independence benchmark `Lambda_indep(u, r) = p(u)/(1-p(u))^r`. The purpose of the functional is not to replace richer theories of tail dependence, but to convert clustering information into a scalar summary that can be compared alongside incidence and volatility measures.

### Proposition set

1. **Monotone ordering by clustering**
   - for equal exceedance probability, a process with smaller clustering coefficient has larger cluster-adjusted stress.

2. **Independent benchmark**
   - if exceedances are temporally independent, then $\theta^{(r)}(u)=(1-p(u))^r$ and $\Lambda_{\mathrm{indep}}(u,r)=p(u)/(1-p(u))^r$.
   - along a high-threshold sequence with $p(u_n)\to 0$, this benchmark converges to the simpler asymptotic reference because $\theta_n^{(r)}\to 1$.

3. **Relative plug-in consistency**
   - under intermediate thresholds, if the exceedance-rate and finite-level clustering estimators are relatively consistent and the denominator stays away from zero, then $\widehat{\Lambda}_n/\Lambda_n \to_p 1$.
   - this is a continuous-mapping corollary once the ingredient consistency results are imported from the literature rather than proved in the paper.

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
- Return-time diagnostics against the exponential null:
  - logistic benchmark observable: `249` return times, normalized CV `0.795`, KS distance `0.221`, block-bootstrap p-value `0.056`
  - clustered synthetic stress: `149` return times, normalized CV `1.286`, KS distance `0.168`, block-bootstrap p-value `0.339`
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

The synthetic evidence supports a narrow but meaningful claim. Recurrence diagnostics separate benchmark processes that volatility alone would describe too coarsely, but the return-time object has to be read against a null rather than admired for having a wide spread. In the updated artifact set, the logistic benchmark observable at threshold quantile `0.95` yields `249` return times and a normalized coefficient of variation of `0.795`, which is broadly consistent with an exponential waiting-time benchmark once finite-sample uncertainty is acknowledged. The clustered synthetic stress benchmark, constructed from absolute GARCH(1,1) values at the same threshold quantile, yields `149` return times, a normalized coefficient of variation of `1.286`, a median waiting time of `10.0`, and a 90th percentile of `53.6`. The corresponding QQ comparison shows exactly the pattern the paper should emphasize: excess short waiting times together with a stretched upper tail. The point is not that recurrence replaces standard econometric diagnostics, but that it measures a dimension of stress organization that those diagnostics do not directly encode.

## Threshold robustness and uncertainty

Threshold-based recurrence diagnostics carry two uncertainties at once: ordinary sampling uncertainty and design uncertainty induced by the threshold and run-length choices. That is why threshold robustness cannot be treated as a stylistic appendix item. The code now reflects that point directly by pairing the threshold scan with a moving-block bootstrap interval rather than leaving the figure as a point-estimate line.

The existing threshold-sensitivity artifact already shows the issue clearly. For the benchmark observable used in the scan, the runs-based clustering estimate is `0.888` at the `0.90` quantile with `500` exceedances, `0.909` at the `0.93` quantile with `350` exceedances, and `1.000` from the `0.95` quantile onward as the exceedance count falls from `250` to `50`. The estimate therefore appears more stable at higher thresholds precisely when the effective sample is shrinking most aggressively.

That pattern gives the uncertainty section its main point, but only once the honest finite-level benchmark is included. A single clustering estimate is not enough. The updated artifacts now show both the threshold path for `theta` and the threshold path for the cluster-adjusted stress functional together with their i.i.d. references and block-bootstrap uncertainty. For the threshold scan, the `Lambda` path falls from `0.113` to `0.010`, while the corresponding finite-level independence benchmark falls from `0.152` to `0.010`. This is the correct null-calibration reading for the logistic benchmark observable: the path largely tracks the mechanical `(1-p)^r` effect rather than demonstrating excess clustering.

Run-length sensitivity is the next layer of robustness because the declustering rule is itself part of the model of persistence. In the current run-length scan at threshold quantile `0.90`, the cluster-adjusted functional is `0.100` for run lengths `1` and `2`, then rises to `0.113`, `0.149`, and `0.206` for run lengths `4`, `6`, and `8`. The corresponding bootstrap intervals are `[0.0998, 0.1006]`, `[0.1002, 0.1012]`, `[0.1115, 0.1174]`, `[0.1445, 0.1591]`, and `[0.1968, 0.2213]`. The substantive point is that a longer quiet-period requirement mechanically increases the measured stress load by merging more exceedances into common episodes. That is not a flaw in the diagnostic; it is the reason the run-length choice must be reported explicitly.

### Compact robustness summary

The manuscript now includes a compact robustness table so the main numerical results are visible in one place rather than scattered across figures and running prose. The table reports the threshold-path estimates for `Lambda` at `q=0.90, 0.93, 0.95, 0.97, 0.99` together with their finite-level independence references and bootstrap intervals, and the run-length-path estimates at `r=1, 2, 4, 6, 8` under fixed threshold quantile `0.90`. The threshold side shows that the benchmark observable mostly tracks the null reference as the cutoff rises. The run-length side now makes the key correction explicit: raw monotonicity in `r` is mechanically present even under i.i.d. data, so only the gap to `Lambda_indep` can be read as evidence about clustering.

## Small empirical macro-financial panel

The empirical illustration should no longer be framed as a toy example. The repo now supports a small but real monthly panel built from three FRED series: the civilian unemployment rate, the Moody's Baa minus 10-year Treasury spread, and the Kansas City Financial Stress Index. After alignment on the common sample, the panel runs from February 1990 through May 2026 and contains `435` observations. To keep the comparison with the synthetic section clean, the baseline empirical specification uses upper-tail exceedances at the `0.90` quantile and a runs declustering parameter of `r = 3`. For unemployment, the section should also report a `12`-month difference because the level is dominated by slow regime shifts and does not credibly satisfy the stationarity-and-mixing conditions at the level the theory section assumes.

At the baseline specification, the level-unemployment row has to be downgraded to descriptive status. Its threshold is `8.2`; it has `42` exceedances but only `2` runs clusters, giving `theta = 0.048` and `Lambda = 2.028`. That is mostly a formal restatement of two long recessionary episodes, not a serious inference target. The transformed unemployment series is the one to use for comparison: with a `12`-month difference, the threshold is `1.30`, there are still `42` exceedances but now `4` clusters, `theta = 0.095`, and `Lambda = 1.043`. The Baa-Treasury spread remains less concentrated by point estimate, with threshold `3.16`, `43` exceedances, `8` clusters, `theta = 0.186`, and `Lambda = 0.531`. KCFSI is similar: threshold `1.068`, `44` exceedances, `8` clusters, `theta = 0.182`, and `Lambda = 0.556`. The substantive claim therefore becomes narrower: labor-market stress still appears more concentrated than the financial indicators, but the level-series headline is descriptive only and the transformed comparison is much less dramatic.

The uncertainty treatment also needed a real fix. The repaired code now conditions the bootstrap on the original numeric threshold and uses basic reflected intervals instead of recycling percentile bands that mixed threshold re-selection into fixed-design uncertainty. Rows with fewer than `3` clusters are now flagged as uninterpretable instead of being assigned a confidence interval. That means the level-unemployment row should report `interval not interpretable, k clusters = 2` for both `theta` and `Lambda`. For transformed unemployment, the `90%` intervals are `[0.014, 0.124]` for `theta` and `[0, 1.853]` for `Lambda`. For the Baa-Treasury spread they are `[0.093, 0.209]` and `[0.189, 0.860]`; for KCFSI they are `[0, 0.255]` and `[0, 1.043]`. Interval overlap is substantial, so the corrected section should not say that the ranking ``survives'' in any strong sense. The defensible claim is that the transformed unemployment point estimate remains larger, but the panel evidence is noisy and should be presented that way.

The new empirical robustness figure should still carry much of the exposition, but with a sharper caveat. Along the threshold dimension, level unemployment is the most sensitive series: `Lambda` falls from `2.028` at `q = 0.90` to `0.838`, `0.415`, and `0.130` at `q = 0.93`, `0.95`, and `0.97`. That pattern is descriptive evidence that the broader upper-tail unemployment regime is dominated by a few long episodes. The Baa-Treasury spread and KCFSI decline more smoothly, moving from `0.531` to `0.130` and from `0.556` to `0.150` over the same grid. Along the run-length dimension, level unemployment is essentially invariant because its two stress episodes are already separated by long calm intervals, while the two financial indicators become progressively more clustered as the episode definition widens. The Baa-Treasury spread rises from `0.425` at `r = 1` to `0.607` by `r = 4`; KCFSI rises from `0.371` at `r = 1` to `0.742` by `r = 6`. The corrected empirical reading is therefore conditional: level unemployment looks most concentrated descriptively, but the transformed-and-bootstrapped comparison is materially less sharp.

A small coverage harness now exists for the repaired interval method on the clustered synthetic benchmark. Using a long-run reference with `theta ≈ 0.646` and `Lambda ≈ 0.155`, the nominal `90%` basic bootstrap intervals achieve empirical coverage `0.875` for `theta` and `1.000` for `Lambda` over `40` replications. That is not deep theory, but it is enough to show that the interval construction is no longer failing the elementary sanity checks that motivated the fix.

Joint activation gives the section a clear historical center. If joint stress is defined as at least two of the three indicators exceeding threshold in the same month, the panel contains `29` joint stress months, the mean stress score is `0.099`, and the maximum activation reaches all three indicators simultaneously. The dominant joint cluster runs from August 2008 through August 2009. Smaller clusters appear around the 2001 recession window, the 2011 euro-area stress period, and the first pandemic shock in March through May 2020. None of this should be overstated as a full macro-financial application, but it is enough to show that the recurrence language produces interpretable distinctions once it is applied to real economic series rather than to bundled examples.

That is the proper role of the section. It is still modest, because there is no structural identification and no claim of optimal thresholding, but it is no longer a placeholder. It shows that the framework can separate long labor-market stress blocks from repeated but more episodic credit and financial turbulence, and that this distinction is not a one-threshold artifact.

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
