---
title: Nominal / categorical metrics — interview deep dive
---

# Nominal / categorical metrics — interview deep dive

[← Family page](./index.md) · [← Home](../index.md)

---

## Q1. "Cramér's V vs Theil's U — when each?"

**Answer.** Cramér's V is **symmetric** — `V(X,Y) = V(Y,X)`. Theil's U is **asymmetric** — `U(X|Y)` measures how much knowing Y predicts X. Use Cramér's V for general "are these related" exploration; use Theil's U when direction matters (e.g., feature → target predictive power, where the reverse direction is uninteresting).

---

## Q2. "Why would Cramér's V mislead with small samples?"

**Answer.** χ² has known small-sample bias — it inflates with low expected counts. Cramér's V inherits this. With 5 categories on each axis and 50 samples, expected counts in some cells are ~2 (below the 5-count rule of thumb). Cramér's V can be 0.4 from noise alone. Use **bias-corrected Cramér's V** (the Bergsma-Wicher correction) or filter samples below a threshold.

---

## Q3. "What κ values are acceptable for an annotation project?"

**Answer.** Landis-Koch interpretation: < 0.4 poor, 0.4-0.6 moderate, 0.6-0.8 good, > 0.8 excellent. Practical floor for shippable training data: κ ≥ 0.6 for general labels, ≥ 0.7 for high-stakes (medical, legal). If the team's first round of annotation gets κ = 0.3, fix the **guidelines** before more annotation; you cannot train your way out of bad labels.

---

## Q4. "Cohen's κ vs Fleiss κ — when each?"

**Answer.** Cohen for two raters; Fleiss for three or more. Modelling assumptions differ — Fleiss assumes raters are exchangeable (any subset of N raters per item is fine), Cohen assumes a fixed two-rater pair. For per-pair diagnostics, compute Cohen's pairwise; for the overall metric across the panel, Fleiss.

---

## Q5. "Why is Theil's U better than mutual information for feature selection?"

**Answer.** MI is unbounded and not normalised — depends on the entropy of both variables. Theil's U normalises by `H(target)`, so it's bounded `[0, 1]` and interpretable as "fraction of target uncertainty resolved by this feature." Useful when comparing feature importance across heterogeneous types. (For purely numerical features, mutual info estimators are still preferred.)

---

## Q6. "How do you measure association between a categorical and a numeric variable?"

**Answer.** Not with Cramér's V (both must be categorical). Three options:
1. **Bin the numeric**, then Cramér's V — loses information.
2. **One-way ANOVA F-statistic** — converts to a single test statistic.
3. **η² (eta-squared)** — fraction of variance in the numeric explained by the categorical. Equivalent to R² for ANOVA. Most interpretable.

TorchMetrics doesn't have η² directly; compute via `1 - SS_within / SS_total` after grouping.

---

## Q7. "Fairness audit — Cramér's V on (model output × demographic)?"

**Answer.** Possible but coarse. Tells you "is output correlated with group?" — but Cramér's V doesn't separate **legitimate** correlation (output should differ if base rates differ) from **bias** (output differs *given the truth*). For fairness, use **per-group classification metrics** (TorchMetrics' `BinaryFairness`) and pre-commit to a fairness definition (demographic parity, equalised odds, etc.).

---

## Q8. "What's the most common bug?"

**Answer.** Using ordinary Cramér's V on a 2×2 table and not realising it equals Pearson's φ correlation, just sign-stripped. Reporting both is redundant. For 2×2, Cramér's V == |φ| == |Pearson's r|.

---

[← Back to family page](./index.md)
