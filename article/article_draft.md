# Recurrence Diagnostics for Econometric Stress Episodes

## Working title

**Recurrence Diagnostics for Econometric Stress Episodes: A Methods Paper with a Cluster-Adjusted Stress Functional**

## Core thesis

Standard econometric stress diagnostics usually measure the magnitude of variation, tail losses, or event frequency. They do not directly measure the temporal organization of extremes. That omission matters because two series can have similar volatility or similar exceedance counts while differing sharply in whether extreme observations arrive as isolated shocks or as clustered stress episodes.

The paper argues that recurrence should be treated as a first-class stress object. The contribution is twofold:

1. a serious methods contribution that integrates exceedance incidence, return times, runs-based clustering, multivariate stress activation, and volatility baselines in one reproducible workflow;
2. a narrow theory contribution through a cluster-adjusted stress functional that scales exceedance incidence by clustering intensity.

The claims are intentionally modest. The paper does not present a new asymptotic theory for extremal-index estimation, nor a structural macro-financial model. It provides a mathematically explicit and computationally reproducible bridge between dependent-extremes diagnostics and applied stress analysis.

## Research question

How should econometric stress analysis distinguish between the severity of extreme observations and the temporal organization of their recurrence?

Sub-questions:

1. What objects should be estimated if the analyst cares about repeated stress episodes rather than isolated large observations?
2. How can those objects be integrated with standard volatility diagnostics?
3. Can clustering information be converted into a scalar stress summary that is easy to compare across benchmark processes?

## Literature review structure

### 1. Extreme values under dependence

The starting point is classical extreme-value theory for dependent sequences. Leadbetter, Lindgren, and Rootzen, Embrechts, Kluppelberg, and Mikosch, and Coles establish why tail behavior under dependence cannot be reduced to iid intuition. For the present paper, the key takeaway is that clustering changes the meaning of exceedance counts.

### 2. Extremal index and clustering diagnostics

O'Brien, Ferro and Segers, and Suveges provide the relevant theory and estimation strategies for clustering intensity in exceedance sequences. The paper does not improve those estimators. It uses a runs-based clustering coefficient as a practical ingredient in a recurrence-oriented stress workflow.

### 3. Recurrence and return-time analysis

The recurrence literature in stochastic and dynamical systems shows that return times and hitting times are structurally meaningful objects, not just descriptive curiosities. That literature motivates the use of inter-exceedance times as direct stress diagnostics.

### 4. Volatility econometrics and tail-risk baselines

ARCH and GARCH models remain essential for changing second-moment conditions. Tail-risk and systemic-risk measures such as CoVaR, SRISK, and related capital-shortfall metrics quantify severity and cross-sectional dependence. The paper does not reject those tools. It argues that they answer a different question from recurrence.

### 5. Gap

There is no shortage of theory about extremes, and there is no shortage of econometric tools for volatility and stress measurement. The gap is the lack of a reproducible bridge that makes recurrence, clustering, and return-time diagnostics operational in the same workflow as conventional econometric summaries.

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

- $\widehat{p}_n$ measures how often the process enters the stress region;
- dividing by $\widehat{\theta}_n^{(r)}$ raises the stress load when exceedances are more clustered;
- under independence, the adjustment disappears.

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

The synthetic evidence supports a narrow but meaningful claim:

- recurrence diagnostics separate benchmark processes that volatility alone would describe too coarsely;
- return-time distributions reveal dispersion that an event count cannot summarize;
- multivariate stress activation identifies when stress is joint rather than isolated;
- volatility remains useful, but it is not a substitute for recurrence structure.

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

The paper does **not**:

- derive new asymptotic theory for extremal-index estimation;
- provide threshold-optimality theory;
- deliver a structural macro-financial model;
- identify contagion or causal propagation;
- justify strong empirical claims about real economies from synthetic evidence alone.

The paper **does**:

- formalize recurrence diagnostics for econometric stress analysis;
- integrate those diagnostics with baseline volatility summaries;
- provide a reproducible artifact pipeline;
- add a narrow original theoretical object with interpretable properties.

## What would strengthen the paper further

1. a threshold-robustness section with explicit sensitivity plots for $\widehat{\theta}_n^{(r)}$ and $\widehat{\Lambda}_n(u_n,r)$;
2. uncertainty quantification for the clustering estimates;
3. one carefully defended empirical illustration on real macro-financial data;
4. stronger positioning against existing systemic-risk dashboards and early-warning models.
