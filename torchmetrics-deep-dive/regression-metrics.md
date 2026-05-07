---
title: Regression Metrics
nav_order: 6
---

# Regression Metrics

Regression metrics in TorchMetrics live under `torchmetrics.regression`. They cover error metrics (MAE, MSE, RMSE), goodness-of-fit (R²), correlations (Pearson, Spearman, Kendall), distributional metrics (KL, JS, CRPS), and forecasting-specific ones (MAPE, NRMSE).

---

## The standard error metrics

| Metric | Class | Formula | When to use |
|---|---|---|---|
| Mean Absolute Error | `MeanAbsoluteError` | `mean(|y - ŷ|)` | Outlier-resistant; same units as target. |
| Mean Squared Error | `MeanSquaredError` | `mean((y - ŷ)²)` | Penalizes large errors; differentiable. |
| Root MSE | `MeanSquaredError(squared=False)` | `√MSE` | Same units as target; reports differently than MSE. |
| Mean Abs % Error | `MeanAbsolutePercentageError` | `mean(|y - ŷ| / |y|)` | Forecasting; **breaks if y can be 0**. |
| Symmetric MAPE | `SymmetricMeanAbsolutePercentageError` | `mean(2|y-ŷ| / (|y|+|ŷ|))` | MAPE without the divide-by-zero cliff. |
| Weighted MAPE | `WeightedMeanAbsolutePercentageError` | `sum |y-ŷ| / sum |y|` | More robust than MAPE; standard in retail forecasting. |
| Normalized RMSE | `NormalizedRootMeanSquaredError` | `RMSE / norm(y)` (range/mean/iqr/std) | Cross-series comparability. |
| Log MSE | `MeanSquaredLogError` | `MSE on log(y), log(ŷ)` | Relative-error tasks (e.g. price). |
| Log-cosh | `LogCoshError` | `mean(log(cosh(y - ŷ)))` | Smooth, MAE-like at large residuals. |
| Relative SE | `RelativeSquaredError` | `Σ(y-ŷ)² / Σ(y-ȳ)²` | Standardized version of MSE. |
| Minkowski | `MinkowskiDistance(p=...)` | `(Σ|y-ŷ|^p)^(1/p)` | Generalizes MAE (p=1) and L2 (p=2). |

All are tensor-state metrics — they keep `(sum_error, n)` (or analogous) and reduce by `sum`. They scale to massive eval sets without RAM blow-up.

---

## R² and explained variance

```python
from torchmetrics.regression import R2Score, ExplainedVariance

r2  = R2Score()
ev  = ExplainedVariance()
```

`R2Score` keeps four sufficient statistics (`sum_y`, `sum_y_sq`, `sum_residuals_sq`, `n`) and computes:

```
R² = 1 - SS_res / SS_tot
SS_res = Σ(y - ŷ)²
SS_tot = Σ(y - ȳ)²
```

It supports `multioutput="uniform_average" | "variance_weighted" | "raw_values"`, matching scikit-learn's API. **Important**: R² can be negative (worse than predicting the mean). Don't clip it to `[0, 1]`.

`ExplainedVariance` is the same idea but uses the variance of residuals rather than their sum of squares.

---

## Correlation metrics

| Metric | Class | What it measures | Notes |
|---|---|---|---|
| Pearson | `PearsonCorrCoef` | Linear correlation. | Welford-style streaming algorithm — keeps running mean / variance / cov. |
| Spearman | `SpearmanCorrCoef` | Monotonic correlation via ranks. | List-state (needs full sample to rank). |
| Kendall τ | `KendallRankCorrCoef` | Rank-pair concordance. | List-state; expensive for huge n. |
| Concordance CC | `ConcordanceCorrCoef` | Lin's CCC — agreement, not just correlation. | Tensor-state. |
| Cosine Similarity | `CosineSimilarity` | `(y · ŷ) / (||y|| ||ŷ||)` | Useful for embeddings as a regression target. |

Pearson is one of the most beautifully implemented files in the repo — it uses the numerically-stable streaming covariance update so you don't need to keep all data.

Spearman and Kendall are inherently list-state because rank statistics require the full population. Use `compute_on_cpu=True` for large eval sets.

---

## Distributional / probabilistic metrics

| Metric | Class | Use |
|---|---|---|
| KL divergence | `KLDivergence` | Compare two probability distributions. |
| JS divergence | `JensenShannonDivergence` | Symmetric KL. |
| CRPS | `ContinuousRankedProbabilityScore` | Score continuous probabilistic forecasts (weather, finance). |
| Critical Success Index | `CriticalSuccessIndex` | Forecast verification (TP / (TP+FP+FN)). |

CRPS is the regression analog of log-loss for probabilistic forecasts. Pair it with `QuantileLoss` (which lives under `regression` in some versions) when you train pinball-loss quantile networks.

---

## Practical advice

1. **Pick the loss-aligned metric**. Train with MSE, report RMSE (same units). Train with MAE, report MAE.
2. **Don't trust MAPE blindly**. Zeros and small values blow it up. Prefer SMAPE or wMAPE in production forecasting.
3. **Multi-output regression** — pass `num_outputs` and use `multioutput="raw_values"` to debug per-target performance.
4. **Streaming vs. list state** — almost all regression metrics are tensor-state, so they scale linearly in input size. Spearman / Kendall are the exceptions.
5. **When you also need confidence**, wrap with `BootStrapper(metric, num_bootstraps=...)` to get a CI.

---

## Reading the source

`MeanAbsoluteError` is a perfect "first metric to read":

```python
class MeanAbsoluteError(Metric):
    is_differentiable = True
    higher_is_better  = False
    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum_abs_error", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total",         torch.tensor(0),   dist_reduce_fx="sum")

    def update(self, preds, target):
        abs_err = (preds - target).abs()
        self.sum_abs_error += abs_err.sum()
        self.total         += abs_err.numel()

    def compute(self):
        return self.sum_abs_error / self.total
```

Three patterns from this file generalize:

1. **Sufficient statistics**: keep `sum` and `count` separately, divide in `compute()`.
2. **Tensor states with `dim_zero_sum` reduction**: works perfectly across DDP.
3. **`is_differentiable = True`**: MAE can be used inside a loss; gradients flow through `compute()`.
