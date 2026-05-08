---
title: Nominal / categorical association metrics — deep dive
---

# Nominal / categorical association metrics — deep dive

> "Are these two categorical variables related?" Used in EDA, feature selection, fairness audits, and inter-rater reliability.

[← Home](../index.md) · [Interview deep-dive →](./interview-deep-dive.md)

---

### CramersV (`tm.nominal.CramersV`)

**What it computes.** `V = sqrt(χ² / (n · min(r-1, c-1)))` where χ² is the chi-square statistic over a contingency table.

**Range / direction.** `[0, 1]`. Higher = stronger association.

**Real-world scenario.** Categorical-vs-categorical correlation matrix in EDA. Pearson correlation requires numeric; Cramér's V handles `device_type ∈ {iOS, Android, Web}` vs `country`. Standard for "is country related to device?" feature exploration.

**Pitfalls.**
- Biased upward for small samples / many categories. Use **bias-corrected Cramér's V** for n < 100 or category count > 5.

---

### TschuprowsT (`tm.nominal.TschuprowsT`)

**What it computes.** `T = sqrt(χ² / (n · sqrt((r-1)(c-1))))`.

**Intuition.** Variant of Cramér's V using a different normalisation. Gives the same value when the table is square; different for non-square.

---

### PearsonsContingencyCoefficient (`tm.nominal.PearsonsContingencyCoefficient`)

**What it computes.** `C = sqrt(χ² / (χ² + n))`.

**Pitfalls.**
- Maximum of `C` is *less than 1* and depends on table dimension, making cross-table comparisons unsafe. Cramér's V is the more standard choice for that reason.

---

### TheilsU (`tm.nominal.TheilsU`)

**What it computes.** `U(X|Y) = (H(X) − H(X|Y)) / H(X)`. Information gain about X from knowing Y, normalised by H(X).

**Intuition.** Asymmetric: `U(X|Y) ≠ U(Y|X)`. "How much does Y predict X?" — directional.

**Range / direction.** `[0, 1]`. Higher better.

**Real-world scenario.** Feature importance for categorical features: `U(target | feature)` measures predictive power. Asymmetric is the right semantics — features predict targets, not the other way around.

---

### FleissKappa (`tm.nominal.FleissKappa`)

**What it computes.** Inter-rater agreement for **multiple** raters. Generalises Cohen's κ (which is two raters).

**Range / direction.** `[-1, 1]`. Higher better; 0 = chance, 1 = perfect.

**Real-world scenario.** Annotation campaigns: 5 annotators label the same 1000 images. Fleiss κ measures their agreement. Common interpretation: κ < 0.4 poor, 0.4-0.6 moderate, 0.6-0.8 good, > 0.8 excellent. If κ < 0.4, the labelling guidelines need to be tightened *before* training.

---

## Quick-reference

| Scenario | Primary | Secondary |
|---|---|---|
| Categorical correlation matrix | Cramér's V | Theil's U (directional) |
| Asymmetric feature relevance | Theil's U | mutual info |
| Multi-annotator agreement | Fleiss κ | Cohen's κ pairwise |
| Two-annotator agreement | Cohen's κ | Fleiss κ |
| Small sample contingency | bias-corrected Cramér's V | Pearson's C |

---

[Continue → interview deep-dive](./interview-deep-dive.md)
