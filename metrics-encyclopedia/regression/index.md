---
title: Regression metrics — deep dive
---

# Regression metrics — deep dive

> Every regression metric is a different *loss surface* on the residual `(y − ŷ)`. Choosing wrong silently changes which models look best — particularly with outliers, scale, and skew.

[← Home](../index.md) · [Interview deep-dive →](./interview-deep-dive.md)

## The residual lens

Define `e_i = y_i − ŷ_i` (signed) and `|e_i|` (absolute). Almost every metric on this page is one of:

- **Power-mean of `|e|`**: `(mean |e|^p)^{1/p}`. p=1 → MAE, p=2 → RMSE, p→∞ → max error.
- **Relative**: divide by `|y|` (MAPE) or by some scale (MASE, RSE).
- **Correlation-based**: how *ranked* are residuals (Pearson, Spearman, Kendall).
- **Distributional**: KL/JS/CRPS — comparing distributions, not point estimates.

Three traps before you pick anything:
1. **Outliers** — RMSE squares them. MAE doesn't. One bad data point can flip the ranking.
2. **Scale invariance** — comparing across products with different price ranges? Use a relative metric.
3. **Skew** — log-transform the target. Now MAPE and R² mean different things than before.

---

## Power-mean residual metrics

### MeanAbsoluteError (MAE) (`tm.regression.MeanAbsoluteError`)

**What it computes.** `(1/N) Σ |y_i − ŷ_i|`.

**Intuition.** Average size of the mistake, in the *units of y*. Robust to outliers (linear penalty).

**Range / direction.** `[0, ∞)`. **Lower better.** Same units as the target.

**When to use.** When errors are roughly equally costly regardless of magnitude. Forecasting demand, ETA prediction, anything reported in user-facing units.

**When NOT to use.** When large errors are disproportionately bad (compute use RMSE) or relative scale matters (use MAPE/sMAPE/wMAPE).

**Real-world scenario.** ETA prediction for ride-sharing: a 10-minute mistake is twice as bad as a 5-minute mistake — linear cost. RMSE would over-weight a single 60-minute outlier; MAE keeps the average honest.

**Code.**
```python
from torchmetrics.regression import MeanAbsoluteError
mae = MeanAbsoluteError()
mae(preds, target)
```

**Pitfalls.**
- "MAE = 5" with targets in seconds and minutes ≠ same model.  Always co-report units.
- MAE on a skewed target (lognormal demand) under-weights tail errors that the business cares about.

---

### MeanSquaredError (MSE) (`tm.regression.MeanSquaredError`)

**What it computes.** `(1/N) Σ (y_i − ŷ_i)²`.

**Intuition.** Quadratic penalty — large errors hurt disproportionately. Same loss as L2 regression.

**Range / direction.** `[0, ∞)`. **Lower better.** Units = target².

**When to use.** When training/optimization. Differentiable everywhere with simple gradient.

**When NOT to use.** As a *reporting* metric — units (target²) are unintuitive. Use RMSE.

**Real-world scenario.** Training loss for a wind-power forecasting model. RMSE for the report.

---

### MeanSquaredError(squared=False) → RMSE

**What it computes.** `sqrt(MSE)` — same units as target.

**Intuition.** Outlier-sensitive average error. Most popular regression KPI.

**Range / direction.** `[0, ∞)`. **Lower better.**

**Real-world scenario.** Weather temperature forecasting: RMSE in °C is the meteorological standard because squared errors penalize forecast misses that produce big public-impact decisions (heat warnings).

**Pitfalls.**
- RMSE is `>= MAE` always (Jensen). Big gap → outliers / heteroscedasticity.

---

### NRMSE — Normalized RMSE (`tm.regression.NormalizedRootMeanSquaredError`)

**What it computes.** `RMSE / norm`, where norm is `mean(y)`, `range(y)`, `std(y)`, or `IQR`.

**Intuition.** Scale-invariant RMSE. Lets you compare across products / units.

**Range / direction.** `[0, ∞)`. Lower better.

**Real-world scenario.** Cross-store demand-forecast leaderboard: each store has different scale. NRMSE-by-mean lets you rank models with one number.

**Pitfalls.**
- `norm="range"` is fragile to outliers in y. `norm="iqr"` is the robust choice.

---

### LogMSE / LogCoshError (`tm.regression.LogCoshError`)

**LogMSE.** `(1/N) Σ (log(1+y) − log(1+ŷ))²` — operates in log-space.

**LogCosh.** `(1/N) Σ log(cosh(y − ŷ))`. Smooth, asymptotically MAE for large errors and MSE for small ones.

**Real-world scenario (LogMSE).** Demand forecasting over multi-order-of-magnitude products: 10 vs 100 should hurt less than 1000 vs 10000 in linear terms but equally in relative terms. Log-space achieves that.

**Real-world scenario (LogCosh).** Robust regression training loss without picking δ for Huber.

---

### Min/Max relative metrics

### MeanAbsolutePercentageError (MAPE) (`tm.regression.MeanAbsolutePercentageError`)

**What it computes.** `(1/N) Σ |y − ŷ| / |y|`.

**Intuition.** Average percent error. Scale-invariant.

**Range / direction.** `[0, ∞)`. Lower better. Often reported as a percent (×100).

**When to use.** Forecasting reported in % terms (revenue, demand, traffic).

**When NOT to use.** Targets near zero (division blows up). Asymmetric: under-forecast capped at 100%, over-forecast unbounded.

**Real-world scenario.** Quarterly revenue forecast for a finance dashboard — "we're 4% off on average" is the natural way to communicate it.

**Pitfalls.**
- Targets at zero → infinite contribution. TorchMetrics adds an epsilon — but the contribution is still huge. Filter zeros explicitly or use sMAPE.
- Asymmetry rewards under-forecasting; teams gaming MAPE shrink predictions.

---

### SymmetricMAPE / sMAPE (`tm.regression.SymmetricMeanAbsolutePercentageError`)

**What it computes.** `(1/N) Σ 2|y − ŷ| / (|y| + |ŷ|)`.

**Intuition.** MAPE made symmetric — over-forecast and under-forecast hurt equally.

**Range / direction.** `[0, 2]` (or `[0, 200%]`). Lower better.

**Real-world scenario.** Forecast competitions (M3, M4) use sMAPE precisely because plain MAPE rewards "shrink the forecast."

**Pitfalls.**
- sMAPE is *still* undefined when both `y` and `ŷ` are zero. TorchMetrics returns 0 in that case.

---

### WeightedMAPE / wMAPE (`tm.regression.WeightedMeanAbsolutePercentageError`)

**What it computes.** `Σ |y − ŷ| / Σ |y|`. A pooled MAPE.

**Intuition.** Weights each sample's contribution by its own magnitude — high-volume samples matter more.

**Range / direction.** `[0, ∞)`. Lower better.

**Real-world scenario.** Product-demand forecast where one SKU sells 10× more than another. wMAPE makes the high-volume SKU matter more — which is what the business KPI cares about. Simple MAPE treats all SKUs equally, which is the wrong cost model for a P&L.

**Pitfalls.**
- For non-negative targets only. With negatives you need separate handling.

---

### MinkowskiDistance (`tm.regression.MinkowskiDistance`)

**What it computes.** `(Σ |y − ŷ|^p)^{1/p}`. Generalizes L1 (p=1) and L2 (p=2).

**Intuition.** A knob that interpolates between MAE-like and RMSE-like and beyond (max-error in the limit).

**Real-world scenario.** Hyperparameter sweep where you want to tune the *metric*: which p best correlates with downstream KPI?

---

### TweedieDevianceScore (`tm.regression.TweedieDevianceScore`)

**What it computes.** Deviance for Tweedie distribution (compound Poisson-Gamma); a Bregman divergence.

**Intuition.** Right metric when the target is **non-negative with a point mass at 0** — claims, revenue per visit, click-conversion: most are zero, the non-zero ones are a continuous distribution.

**Range / direction.** `[0, ∞)`. Lower better. The `power` argument selects the family (1 = Poisson, 2 = Gamma, 1.5 = compound).

**Real-world scenario.** Insurance claim severity: most policy-holders never claim (zero), the rest claim a continuous amount. Squared error treats the zeros as outliers and biases predictions upward; Tweedie deviance handles the structure correctly.

**Pitfalls.**
- Requires non-negative predictions. Clip with `softplus` or `relu`.

---

### CRPS — Continuous Ranked Probability Score (`tm.regression.ContinuousRankedProbabilityScore`)

**What it computes.** For probabilistic forecasts, the integrated squared error between the CDF of the predicted distribution and the indicator of the truth: `∫ (F(x) − 1[x ≥ y])² dx`.

**Intuition.** Generalises MAE to probabilistic forecasts. Reduces to MAE when the prediction is a Dirac.

**Range / direction.** `[0, ∞)`. Lower better.

**Real-world scenario.** Wind-power probabilistic forecast (quantile regression or ensemble). CRPS is the standard metric in numerical weather prediction; comparing distributions, not point estimates.

**Pitfalls.**
- Needs distribution samples or a CDF — not directly applicable to point predictions.

---

### CSI — Continuous Skill Index (`tm.regression.CSI`)

**What it computes.** A skill score over a baseline forecast, scale-invariant.

**Real-world scenario.** Operational forecasting where the comparison is "are we beating climatology / persistence?" CSI says yes/no in a single number.

---

## Goodness-of-fit and explained variance

### R² / R2Score (`tm.regression.R2Score`)

**What it computes.** `1 − SS_res / SS_tot`. Fraction of variance in `y` explained by predictions.

**Intuition.** "How much better than predicting the mean?"

**Range / direction.** `(-∞, 1]`. **Higher better.** 0 = same as predicting the mean. Negative = worse than mean.

**When to use.** When you want a *relative* quality score. Comparing across datasets with different y-variances.

**When NOT to use.** Tiny test set (R² is high-variance). Highly skewed target (R² rewards models that fit the bulk and ignore tails).

**Real-world scenario.** A house-price model: R² 0.85 says you explain 85% of price variance. Compare across cities (different price scales) directly.

**Pitfalls.**
- **Negative R² is real.** Means your model is worse than predicting the mean. Don't clip to 0.
- "Adjusted R²" exists for tracking complexity; TorchMetrics has it as `R2Score(adjusted=...)`.
- R² is not invariant to log-transform on y; the same model has different R² in raw vs log-space.

---

### ExplainedVariance (`tm.regression.ExplainedVariance`)

**What it computes.** `1 − Var(y − ŷ) / Var(y)`. Like R² but ignores systematic bias.

**Intuition.** R² minus the part driven by `mean error`. Two models with same residual variance but different means → same EV, different R². Use EV when bias is corrected post-hoc.

**Real-world scenario.** Time-series with a constant offset that's calibrated separately (e.g., bias-corrected weather forecast): EV measures the model's *ranking* skill, R² penalises the offset that you'll calibrate away anyway.

---

### RelativeSquaredError (RSE) (`tm.regression.RelativeSquaredError`)

**What it computes.** `Σ (y − ŷ)² / Σ (y − ȳ)²`. The unsubtracted complement of R².

**Intuition.** RSE = 1 − R² (when R² ≥ 0). Lower better.

---

## Correlation metrics

### Pearson (`tm.regression.PearsonCorrCoef`)

**What it computes.** Linear correlation. `(Σ (x − x̄)(y − ȳ)) / (σ_x σ_y)`.

**Intuition.** How linearly related two vectors are. Insensitive to scale and offset.

**Range / direction.** `[-1, 1]`. Higher (in absolute value) better.

**When to use.** When the *ranking* matters more than the values. Sanity-check: if Pearson is 0.99 and your scale-aware metric (RMSE) is bad, you have a *calibration* problem (offset/scale), not a signal problem.

**Pitfalls.**
- Outliers can drive Pearson up *or* down dramatically. Use Spearman for robust signal-strength.

---

### Spearman (`tm.regression.SpearmanCorrCoef`)

**What it computes.** Pearson on ranks. Captures monotonic (not just linear) relationships; robust to outliers.

**Real-world scenario.** Ranking judgements — "is the model preserving the order of items even if the magnitudes are off?" Standard for IR scoring quality.

---

### Kendall's τ (`tm.regression.KendallRankCorrCoef`)

**What it computes.** Concordance between rankings: `(C − D) / total_pairs`.

**Real-world scenario.** Strict pairwise ranking quality on small lists (top-10 results). Kendall is more interpretable than Spearman for small N.

---

### CosineSimilarity (`tm.regression.CosineSimilarity`)

**What it computes.** `(y · ŷ) / (||y||·||ŷ||)`. Direction agreement between vectors.

**Range / direction.** `[-1, 1]`. Higher better.

**Real-world scenario.** Sentence-embedding regression: cosine = 0.91 means model embedding aligns with target embedding direction; magnitude doesn't matter.

---

### ConcordanceCorrCoef (`tm.regression.ConcordanceCorrCoef`)

**What it computes.** Lin's concordance correlation: agreement *and* magnitude (Pearson with a bias correction).

**Real-world scenario.** Method comparison in clinical chemistry — "does our cheap sensor agree *exactly* with the gold-standard"? Pearson measures correlation, CCC measures correlation **with bias penalty**.

---

## Distribution-distance metrics

### KLDivergence (`tm.regression.KLDivergence`)

**What it computes.** `Σ p(x) · log(p(x)/q(x))`. Asymmetric.

**Intuition.** Information lost when q approximates p. KL = 0 iff `p = q`.

**Range / direction.** `[0, ∞)`. Lower better.

**Real-world scenario.** Drift detection: KL between training feature distribution and last-week production distribution. Increase = drift.

**Pitfalls.**
- Asymmetric. `KL(p||q) ≠ KL(q||p)`. Use JS for symmetric.
- Requires `q > 0` everywhere `p > 0`. Smooth or add ε.

---

### JSDivergence (`tm.regression.JensenShannonDivergence`)

**What it computes.** Symmetric KL: `0.5·KL(p || M) + 0.5·KL(q || M)` with `M = (p+q)/2`.

**Range / direction.** `[0, log 2]`. Lower better.

**Real-world scenario.** Same as KL (drift detection) but symmetric and bounded — better default.

---

## Quick-reference: which regression metric for which scenario?

| Scenario | Primary | Secondary |
|---|---|---|
| ETA / time prediction | MAE | RMSE |
| Weather temperature | RMSE | bias |
| Demand forecast (% units) | wMAPE | MAE per SKU |
| Multi-scale demand | NRMSE-IQR or sMAPE | per-store MAE |
| Insurance claims | Tweedie deviance | MAE |
| Probabilistic forecast | CRPS | calibration |
| Ranking quality | Spearman, Kendall | Pearson |
| House prices, balanced | R² | RMSE |
| Sensor agreement | Concordance CC | Pearson |
| Drift detection | JS divergence | KS-test |
| Method-comparison study | Concordance CC | Bland-Altman plot |

---

[Continue → interview deep-dive](./interview-deep-dive.md)
