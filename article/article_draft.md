# From Chaos to Crisis: Rare-Event Recurrence as an Econometric Diagnostic

## Introduction

Econometric discussions of crises usually center on shocks, volatility, breaks, and regime changes. Those are essential objects. But they do not fully describe a practical question that matters in macro-financial stress: does an extreme event arrive as a one-off spike, or does the system keep returning to a dangerous region?

That distinction is central because social and financial damage is often driven less by one large observation than by recurrence and clustering. A single inflation surprise, one quarter of severe stress, or one volatility burst may be absorbed. Repeated returns to the same dangerous region are harder to absorb. They affect expectations, financing conditions, policy credibility, and the distribution of future risk.

This repository is built around that narrower question. It uses rare-event tools associated with dynamical-systems extreme value theory to study exceedances, return times, recurrence, clustering, and joint stress. The claim is deliberately modest. The repository does not claim that observed economies follow literal deterministic laws. It claims only that recurrence structure in rare events can be measured, summarized, and compared in a reproducible way.

## Why recurrence deserves separate attention

Standard econometrics remains the baseline. Autoregressive models summarize persistence. Volatility models describe changing second moments. Break models and regime methods detect structural change. Those tools are useful and remain necessary.

The limitation is not that they are wrong. The limitation is that crisis-relevant tail behavior is often reduced to large observations inside a broader stochastic model, while the pattern of repeated return to a stress region receives less direct attention than it deserves.

For risk management and policy interpretation, that pattern matters. Two processes may have similar marginal tail behavior and still differ sharply in how extremes cluster through time. If one process produces isolated exceedances and another produces repeated tail occupation, the practical meaning of danger is different even when conventional summaries look similar.

## Conceptual translation

The dynamical-systems language is useful because it translates cleanly into applied diagnostic work:

| Dynamical-systems idea | Econometric interpretation |
|---|---|
| observable along an evolving path | measured economic or financial series |
| rare set | crisis or stress region |
| hitting time | time until a stress region is reached |
| return time | time between repeated visits |
| clustering | persistence of extreme episodes |
| extremal index | compact summary of whether extremes arrive in clusters |
| joint recurrence | multivariate stress across variables |

This translation does not require a strong structural claim about the economy. It only requires a willingness to ask whether observed tail events recur in a patterned way that is worth measuring directly.

## What the repository actually implements

The current codebase supports a reproducible rare-event workflow with the following components:

- threshold exceedance extraction;
- runs-based extremal-index summaries;
- return-time and recurrence diagnostics;
- multivariate joint-stress summaries;
- descriptive comparison with baseline volatility-style diagnostics;
- saved article-facing figures with matching source CSV files.

These are the supported outputs. They are sufficient for a computational article about recurrence diagnostics. They are not sufficient for stronger claims such as causal identification, structural macroeconomic estimation, or exact crisis-date prediction.

The article-facing figure bundle already reflects this scope. The stable assets are:

- `01_conceptual_pipeline.png` for the workflow from raw inputs to article evidence;
- `02_synthetic_extremes_comparison.png` for clustered versus more isolated synthetic extremes;
- `03_return_time_distribution.png` for descriptive return-time behavior;
- `04_extremal_index_by_series.png` for cross-series clustering comparison;
- `05_multivariate_stress_timeline.png` for joint-stress timing;
- `06_econometric_vs_recurrence_diagnostics.png` for descriptive comparison between baseline volatility-style summaries and recurrence-oriented diagnostics.

Each of these figures has a matching source CSV, which means the article can point to saved structured outputs rather than to ad hoc plotting state.

## Synthetic evidence as the current strongest support

The strongest fully supported evidence in the repository is synthetic. That is not a weakness if it is stated honestly. Synthetic systems are the cleanest way to show why recurrence structure matters, because they let us compare processes that differ in clustering behavior without pretending that the comparison is already an empirical finding.

The repository includes controlled generators for familiar stochastic baselines and nonlinear benchmark systems. Those workflows support a simple but important point: similar-looking series can differ materially in recurrence and clustering, and that difference appears in exceedance patterns, return times, and extremal-index summaries.

In article terms, that synthetic support is most directly expressed through `02_synthetic_extremes_comparison.png`, which is the clearest visual anchor for the claim that tail behavior differs not only in magnitude but also in recurrence structure.

This is exactly the right use of the synthetic layer. It is not evidence that economies are chaotic maps. It is evidence that recurrence diagnostics detect behavior not fully summarized by means, variances, or one-dimensional tail thickness alone.

## Extremal index as a practical diagnostic

Among the implemented summaries, the extremal index is especially useful. Informally, it distinguishes between extremes that behave more like isolated episodes and extremes that arrive in clusters. That distinction is often more decision-relevant than another small refinement in average behavior.

In a macro-financial context, clustered exceedances matter because they correspond more closely to sustained stress than to isolated bad draws. A regulator, treasury, or central bank may care less about one excursion beyond a threshold than about repeated short-horizon returns to the same stress region.

The value of the extremal index in this repository is therefore operational rather than metaphysical. It is a compact description of whether the tail is episodic or persistent.

The most direct article-facing support for this point is `04_extremal_index_by_series.png`, which summarizes cross-series clustering differences without requiring a stronger structural interpretation than the code currently supports.

## Return times and recurrence summaries

Return-time summaries complement clustering diagnostics. They ask how long the process typically takes to revisit a threshold-defined stress region. Again, the claim must remain careful: these summaries describe recurrence patterns in observed or simulated data; they do not deliver exact forecasting rules for the next crisis date.

Used correctly, they help distinguish between environments where stress regions are rarely revisited and environments where the system returns quickly and repeatedly. That is a meaningful descriptive difference even when it does not identify the cause of the behavior.

That descriptive role is exactly what `03_return_time_distribution.png` supports. It is evidence about recurrence timing patterns, not a device for exact crisis-date prediction.

## Multivariate stress and joint recurrence

The repository also supports multivariate stress summaries. This matters because macro-financial danger often appears through joint pressure rather than through one variable in isolation. A single stressed series may be manageable; simultaneous stress across inflation, spreads, funding proxies, or market indicators is more consequential.

The supported interpretation here is still descriptive. Joint recurrence diagnostics can show that stress appears together across observables more often than an isolated reading would suggest. They do not, by themselves, identify a structural transmission mechanism.

The relevant stable visual here is `05_multivariate_stress_timeline.png`, which supports a joint-stress timing narrative but not a causal account of why the co-movement occurs.

## Empirical use: template, not result

The repository includes an empirical workflow with local data loaders, catalog-based inputs, cached remote-source patterns, and multi-series configuration support. That means the codebase now supports a credible empirical template.

What it does not support yet is a strong article-level empirical claim on its own. The repository can process a real public dataset supplied locally or through a controlled cache workflow, then produce exceedance, recurrence, and clustering diagnostics. But whether those diagnostics justify a substantive empirical claim still depends on the actual dataset, threshold robustness, sample length, and interpretation discipline.

That distinction is important. The empirical layer is best described as ready for application, not as already delivering validated macroeconomic conclusions by itself.

## Relationship to standard econometric baselines

This project is not a rejection of standard econometrics. It is better understood as a complement. Baseline volatility-style and dependence summaries remain useful benchmarks. The repository explicitly keeps descriptive baseline comparisons because recurrence tools should be interpreted against familiar diagnostics rather than in isolation.

The incremental contribution is therefore specific: recurrence-oriented rare-event summaries provide information about tail occupation, clustering, and revisitation that standard baseline summaries may not foreground.

That complementarity is what `06_econometric_vs_recurrence_diagnostics.png` is for. The figure is strongest when used to compare descriptive lenses, not when used to claim that one framework fully dominates the other.

## Limitations

The article should state the limitations plainly.

First, the repository does not establish deterministic laws for observed economies.

Second, it does not provide causal identification or policy attribution.

Third, synthetic benchmarks are methodological evidence, not empirical evidence.

Fourth, threshold choice matters materially, which is why threshold sensitivity belongs in the workflow rather than in a footnote.

Fifth, short samples and non-stationary real data can make rare-event summaries unstable, especially in the tail.

Those limits do not make the framework unhelpful. They define the zone in which it is honest.

## Conclusion

The central contribution of the repository is not a grand theory of crises. It is a reproducible way to ask a sharper descriptive question: when extremes occur, do they behave like isolated events or like repeated returns to dangerous territory?

That question is worth asking because crisis damage is often driven by recurrence and clustering rather than by one large observation alone. The current codebase supports that claim through synthetic comparisons, recurrence diagnostics, multivariate stress summaries, and reproducible article-facing outputs with matching source files.

The right conclusion is therefore disciplined. Rare-event recurrence diagnostics expand the econometric description of danger. They do not replace standard econometrics, and they do not justify causal or deterministic claims. But they do provide a useful additional language for studying how stress returns, how it clusters, and how it appears jointly across observables.
