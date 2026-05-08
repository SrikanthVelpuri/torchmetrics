---
title: Regression metrics — interview deep dive
---

# Regression metrics — interview deep dive

[← Family page](./index.md) · [← Home](../index.md)

---

## Q1. "MAE vs RMSE — when to use which?"

**Answer.** RMSE squares errors, so it's outlier-sensitive: one 60-minute mistake counts 144× a 5-minute mistake. MAE is linear — counts 12×. Use MAE when errors are roughly equally costly per unit (ETA, demand). Use RMSE when large errors are *disproportionately* bad (peak-hour grid load, safety thresholds). Both report in the units of y.

> **F1.1** "If RMSE is much bigger than MAE, what does that tell you?"
>
> **Answer.** Heavy-tailed residuals. Either outliers, or heteroscedasticity (variance grows with |y|). Plot residuals against `|y|` to distinguish. Outliers → robust loss / Huber. Heteroscedasticity → log-transform or model the variance.
>
> > **F1.1.1** "Why specifically RMSE ≥ MAE always?"
> >
> > **Answer.** Jensen's inequality. `sqrt(mean(x²)) ≥ mean(x)` for any non-negative `x`, with equality iff x is constant. So RMSE = MAE only if every residual is the same magnitude.

---

## Q2. "MAPE has issues — name three."

**Answer.**
1. **Targets near zero** blow up — `1e-6` denominator → `1e+6` MAPE contribution.
2. **Asymmetric** — under-forecast capped at 100%, over-forecast unbounded. Models trained on MAPE under-forecast on average.
3. **Doesn't tolerate negatives**.

> **F2.1** "What replaces it?"
>
> **Answer.** Three options: (1) sMAPE (symmetric, bounded at 200%), (2) wMAPE (volume-weighted, removes the zero issue when totals are non-zero), (3) MASE (scale-free, divides by naive-forecast error). Forecasting competitions converged on sMAPE and MASE specifically because of the asymmetry problem.

> **F2.2** "Your team uses MAPE because business asked for it. How do you push back?"
>
> **Answer.** Don't push back on the *report* — calculate and show MAPE. Push back on the *training loss* — train on quantile loss or Tweedie deviance, then evaluate on MAPE. Tell stakeholders: "the model is optimised for unbiased forecasts; we report MAPE as you requested."

---

## Q3. "Why is R² sometimes negative?"

**Answer.** R² = `1 − SS_res / SS_tot`. Negative means `SS_res > SS_tot`, i.e., your residuals are bigger than just predicting the mean would give. Means the model is *worse* than the trivial baseline. Common causes: (a) trained on a different distribution, (b) wrong sign somewhere, (c) overfit to a feature that's flipped in the test set.

> **F3.1** "Should you clip negative R² to 0 in dashboards?"
>
> **Answer.** **No.** Negative R² is informative — it tells you the model is broken. Clipping hides the failure mode. Show the negative number; investigators will know it means "worse than mean."

---

## Q4. "Why is Pearson correlation insufficient for measuring forecast quality?"

**Answer.** Pearson is invariant to scale and offset. A forecast that's exactly 2× the truth or shifted by +10 has Pearson = 1.0. The forecast is wrong but Pearson says perfect. Use **Lin's concordance correlation** instead — it adds a bias penalty so identical-magnitude is required for CCC = 1.

> **F4.1** "What about Spearman?"
>
> **Answer.** Spearman is Pearson on ranks — captures monotonic relationships, robust to outliers. Useful as a *complement* to Pearson, not a replacement: Spearman = 1 with Pearson = 0.5 means "ranks agree but the slope is wrong."

---

## Q5. "Insurance claim severity — what loss?"

**Answer.** Tweedie deviance with `power = 1.5` (compound Poisson-Gamma). The target has a point mass at zero (most policies don't claim) and a continuous skewed distribution above zero. Squared error treats the zeros as outliers; Tweedie handles the structure. For training, use Tweedie loss; for reporting, MAE per claimant + frequency separately.

> **F5.1** "Why not just split the problem into a Bernoulli + Gamma?"
>
> **Answer.** That's the *frequency × severity* approach and it's also valid. Tweedie unifies them in one loss; the frequency-severity decomposition gives interpretability. Pick: unified Tweedie if you want one model, two-stage if downstream needs the components separately.

---

## Q6. "What's CRPS and when do you use it?"

**Answer.** Continuous Ranked Probability Score. For a probabilistic forecast (a distribution, not a point), CRPS = `∫ (F(x) − 1[x ≥ y])² dx`. Generalises MAE to distributions: a Dirac at `ŷ` gives `CRPS = |y − ŷ|`. Use it for any system that emits a distribution — quantile forecasts, ensemble forecasts, mixture-density networks.

> **F6.1** "How does CRPS interact with calibration?"
>
> **Answer.** CRPS is *strictly proper*: minimised in expectation when the predicted distribution = true distribution. So CRPS rewards good calibration *and* sharpness simultaneously. A flat distribution (high spread, well-calibrated) loses to a sharp distribution (narrow, well-calibrated). Track CRPS + spread separately to see which is dominating.

---

## Q7. "How do you measure forecast quality across products with very different scales?"

**Answer.** Three options, in order of preference:
1. **wMAPE** — volume-weighted, what business cares about.
2. **NRMSE / IQR-normalised** — RMSE divided by the IQR of the target. Scale-free.
3. **MASE** — error divided by the naive forecast (last-period). Scale-free, comparable across series.

Plain MAE summed across products is dominated by the high-volume products. Plain MAPE is dominated by the low-volume noisy ones. wMAPE finds the middle.

---

## Q8. "Drift detection — KL or JS?"

**Answer.** JS, by default. KL is asymmetric and unbounded — `KL(P||Q)` and `KL(Q||P)` give different numbers, neither is comparable across pairs. JS is symmetric, bounded by `log 2`, and reduces to half of KL when distributions are identical. KL is fine when you have a clear "reference" distribution and want directional drift; otherwise JS.

> **F8.1** "What if both have a hole — q(x)=0 where p(x)>0?"
>
> **Answer.** KL is undefined (∞). Add a small smoothing constant (Laplace/ε) to q. JS is fine because the mixture `M = (P+Q)/2` is non-zero everywhere either is.

---

## Q9. "Reporting RMSE = 0.05 — what's missing?"

**Answer.** Units, scale, and a comparison. "RMSE = 0.05 dollars" on a target with mean 1 dollar is bad; "RMSE = 0.05 dollars" on a target with mean 100 dollars is excellent. Always co-report:
- Mean / median of `y` for context.
- Naive baseline (last-value forecast or mean).
- A relative metric (NRMSE or wMAPE).

---

## Q10. "Why does TorchMetrics have so many regression metrics?"

**Answer.** Because no single one captures every cost surface. Outliers, skew, scale, distributions, ranking — each is a different lens. The library exposes them all so the practitioner picks the right one. The interview signal: candidates who only know "MAE/MSE/R²" have a thin toolbox; senior candidates know which lens for which problem.

---

## Q11. "How does TorchMetrics handle DDP for regression metrics?"

**Answer.** Almost all regression metrics keep running counters: `(sum_of_errors, sum_of_squared_errors, n)` etc. `_sync_dist` is `all_reduce(SUM)` — O(1) bandwidth, exact. Pearson/Spearman need the full data (rank correlation needs the global ranks), so those gather the lists; cost O(N).

> **F11.1** "If gathering Pearson is O(N), how do you scale to 100M samples?"
>
> **Answer.** Two options: (1) compute Pearson in moment form: `cov(x,y) = E[xy] − E[x]E[y]`, all of which are sufficient statistics that aggregate as counters → O(1) sync. TorchMetrics does this. (2) for Spearman, sub-sample — Spearman on 50k random samples is unbiased for the population.

---

## Q12. "ETA model — RMSE = 4 minutes, MAE = 2 minutes, p99 = 30 minutes. Ship?"

**Answer.** Ship is a business decision; the metrics tell you the model is mostly accurate but has a heavy tail. p99 = 30 minutes is what users feel — long ETAs that turn into much longer ones erode trust. Two follow-ups before shipping: (a) what fraction of trips hit p99 — is it 1% of high-revenue trips? (b) is the tail on a slice (rush hour, airport drop-offs)? Consider per-segment metrics and a separate tail-targeted model.

---

## Q13. "Probabilistic forecast — what's the right loss?"

**Answer.** **Quantile loss** (pinball) for training: `L_τ(y, ŷ_τ) = max(τ(y − ŷ_τ), (τ−1)(y − ŷ_τ))`. Train on multiple quantiles. Evaluate on:
- **Coverage** at the prescribed level (does the 90% PI contain 90% of truth?)
- **CRPS** on the predicted distribution
- **Sharpness** (mean width of the PI)

Calibration + sharpness is the right framing. A wide PI is well-calibrated but useless; a narrow miscalibrated PI is misleading.

---

## Q14. "What is heteroscedasticity and what do you log?"

**Answer.** When residual variance depends on x or y. Diagnostic: plot |residual| vs `ŷ`. If there's a trend (variance grows with `ŷ`), homoscedastic-loss training gives biased models — small-y points get over-fit, large-y under-fit.

Logs: bin residuals by `ŷ`, compute per-bin RMSE. Plot. Or model `Var(y|x)` directly with a heteroscedastic loss (Gaussian NLL with predicted σ).

---

## Q15. "Concordance correlation — when does it matter?"

**Answer.** Method-comparison studies (clinical, sensor calibration). Two devices measure the same thing; you want them to agree *exactly*, not just rank consistently. Pearson rewards correlated-but-biased devices; CCC penalises bias and slope. The Bland-Altman plot is the visual analog.

---

## Q16. "Forecast bias — what metric and why does it matter?"

**Answer.** Mean signed error: `(1/N) Σ (ŷ − y)`. Reports systematic over/under-prediction in y units. Crucial because:
- Inventory: under-forecast ⇒ stockouts; over-forecast ⇒ holding cost.
- Risk: under-forecast risk ⇒ exposure; over-forecast ⇒ over-hedging.

Always co-report MAE/RMSE *and* signed bias. A model with MAE = 100 and bias = +95 is mostly biased — fix calibration first.

---

## Q17. "How do you compare two regression models on the same dataset rigorously?"

**Answer.** Three layers:
1. **Point estimates** of MAE/RMSE/R² with bootstrap CIs (`BootStrapper`).
2. **Paired statistical tests** on per-sample errors: Diebold-Mariano for time series, paired t / Wilcoxon for IID.
3. **Per-segment** comparison — overall identical, but per-segment one model wins consistently.

A bare "model A's RMSE is 0.04 lower than model B" without CIs is meaningless if the population CI is 0.10.

---

## Q18. "Log-transform the target — what changes?"

**Answer.** Loss in log-space penalizes relative error, not absolute. R² in log-space ≠ R² in linear-space. Predictions need un-transforming for reporting; a naive `exp(ŷ)` is biased low (Jensen's inequality on the convex `exp`); use the right correction (Duan's smearing or moment matching).

---

## Q19. "What's the simplest baseline for any regression problem?"

**Answer.** Naïve forecast (last-period or mean). Compute its MAE/RMSE on your test set. If your model can't beat naïve, you have nothing. This is what MASE bakes in: errors normalised by the naïve baseline. Always report MASE or "MAE relative to naïve" alongside MAE.

---

## Q20. "Quick — what's wrong with reporting only Pearson for a regression model?"

**Answer.** Pearson hides scale and offset bugs. Pearson = 0.99 with `ŷ = 2y + 5` is a model with no predictive value once unbiased. Co-report scale-aware (RMSE/MAE) and a calibration check (intercept of ŷ vs y regression).

---

[← Back to family page](./index.md)
