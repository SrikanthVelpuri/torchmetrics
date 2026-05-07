---
title: Cheat Sheets
nav_order: 21
---

# Visual Cheat Sheets

Dense, scan-able one-pagers. Print, pin, glance. Each section is designed to fit on one screen / one sheet of paper.

---

## 1. The lifecycle (the only diagram you must memorize)

```text
                ┌──────────────────┐
                │   __init__()      │  add_state(name, default,
                │   declares state  │   dist_reduce_fx, persistent)
                └────────┬──────────┘
                         │
                         ▼
   ┌─────────────────────────────────────────────────┐
   │           PER BATCH (called many times)         │
   │                                                  │
   │   ┌─────────────┐         ┌─────────────────┐   │
   │   │  update()   │         │  forward()      │   │
   │   │  (mutate    │  OR     │  = update()     │   │
   │   │   state)    │         │  + per-batch    │   │
   │   │  no return  │         │  compute()      │   │
   │   └─────────────┘         └─────────────────┘   │
   │                                                  │
   └─────────────────────┬────────────────────────────┘
                         │ (epoch end)
                         ▼
                ┌──────────────────┐
                │  compute()       │  sync state across DDP,
                │  pure: state→val │  cache result
                └────────┬──────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  reset()         │  state ← deep-copied defaults
                └──────────────────┘
```

**5 methods. Memorize this and you're 30 % done.**

---

## 2. State + reduction cheat sheet

| State default | Reduction | Use case |
|---|---|---|
| `tensor(0)` | `"sum"` | Counts, totals (TP/FP/N) |
| `tensor(0.0)` | `"sum"` | Sums of errors (MAE, MSE) |
| `tensor(0.0)` | `"mean"` | Already-averaged values |
| `tensor(±inf)` | `"min"` / `"max"` | Best-so-far trackers |
| `[]` | `"cat"` | Full predictions/targets (AUROC, mAP, BLEU) |
| anything | `None` | No reduction — stack across ranks |
| anything | callable | Custom reduction |

**Mantra**: *"Tensor states for counts and sums — list states for distributions."*

---

## 3. Classification metric chooser

```text
Is the task...?
  │
  ├── 1 of 2 mutually exclusive   → BINARY
  │     │
  │     └── Need threshold-free?  → BinaryAUROC, BinaryAveragePrecision
  │     └── Have a threshold?     → BinaryAccuracy, BinaryF1Score
  │     └── Care about calibration? → BinaryCalibrationError
  │     └── Hard precision floor? → BinaryRecallAtFixedPrecision
  │     └── Hard recall floor?    → BinaryPrecisionAtFixedRecall
  │
  ├── 1 of K mutually exclusive   → MULTICLASS
  │     │
  │     ├── Balanced?             → MulticlassAccuracy
  │     ├── Imbalanced?           → MulticlassF1Score(average='macro')
  │     ├── Per-class diagnosis?  → ClasswiseWrapper(... average=None)
  │     └── Calibration?          → MulticlassCalibrationError
  │
  └── Subset of K labels          → MULTILABEL
        │
        ├── All-correct match?    → MultilabelExactMatch (brutal)
        ├── Per-label hit rate?   → MultilabelHammingDistance (kind)
        └── Per-label F1?         → MultilabelF1Score(average='macro')
```

---

## 4. Regression metric chooser

```text
What scale do you want?
  │
  ├── Same units as target  → MAE, RMSE
  ├── Squared (penalty)     → MSE
  ├── Percent               → SMAPE, wMAPE  (NOT MAPE if y can be 0)
  └── Goodness-of-fit       → R2Score, ExplainedVariance
                              (R² can be NEGATIVE — don't clamp!)

Need correlation?
  │
  ├── Linear                → PearsonCorrCoef         (streaming, tensor-state)
  ├── Monotonic             → SpearmanCorrCoef        (list-state)
  ├── Rank concordance      → KendallRankCorrCoef     (list-state, slow)
  └── Agreement (not just corr) → ConcordanceCorrCoef

Need probabilistic / forecasting?
  │
  ├── Continuous distribution score  → CRPS
  ├── Asymmetric cost                → Quantile/Pinball loss (custom)
  └── Forecast verification          → CriticalSuccessIndex
```

---

## 5. Imbalance survival kit

```
Positive rate < 1 %        →  Average Precision  (NOT AUROC alone)
Positive rate < 0.1 %      →  AP + LogAUC + Recall@FixedPrecision
Hard precision constraint  →  Recall@FixedPrecision(min_precision=0.99)
Hard recall constraint     →  Precision@FixedRecall(min_recall=0.95)
Calibrated probability     →  CalibrationError + temperature scaling
```

Why? **AUROC is dominated by the negative class.** AP integrates over recall on positives — the right axis when positives are rare.

---

## 6. The "always pair these" table

| Headline metric | Always also report | Why |
|---|---|---|
| Accuracy | F1 (macro) | Catches imbalance |
| AUROC | Average Precision | Catches imbalance |
| F1 (binary) | Precision + Recall | Decompose to interpret |
| F1 (macro) | F1 (weighted) + Per-class | Catches long-tail |
| MAE | RMSE | Outlier sensitivity |
| MAPE | wMAPE + SMAPE | MAPE explodes near 0 |
| BLEU | chrF or BERTScore | Lexical + semantic |
| FID | KID | Small-N bias check |
| WER | CER + per-domain WER | Granularity + fairness |
| NDCG | MRR + Recall@k | Top-1 + position-sensitivity |

---

## 7. DDP "is my metric correct?" checklist

```
[ ] Every state declared via add_state(...)
[ ] Every state's default is Tensor or empty list (NOT Python list of floats)
[ ] Every state has explicit dist_reduce_fx (not the default None)
[ ] Reductions are associative (sum/mean/cat/min/max)
[ ] update() does NOT call torch.distributed directly
[ ] update() does NOT return anything
[ ] compute() is a pure function of declared states
[ ] full_state_update is set deliberately (True or False, not None)
[ ] Tested at world_size=2 against single-process
```

---

## 8. The four `dim_zero_*` reductions

```python
dim_zero_sum  (x)  →  x.sum(dim=0)
dim_zero_mean (x)  →  x.mean(dim=0)
dim_zero_max  (x)  →  x.max(dim=0).values
dim_zero_min  (x)  →  x.min(dim=0).values
dim_zero_cat  (x)  →  torch.cat(x, dim=0)   # only valid for list states
```

**Mnemonic**: *"SUM, MEAN, MAX, MIN, CAT — five flavors of stack-and-reduce."*

---

## 9. The two `forward()` paths

| Property | `_forward_full_state_update` (safe) | `_forward_reduce_state_update` (fast) |
|---|---|---|
| When picked | `full_state_update=True` or `None` | `full_state_update=False` |
| update() calls per forward | **2** | **1** |
| Restores global state via | Save/restore | Reduction merge |
| Works for any metric? | Yes | Only if state is reducible by registered fx |
| Speed | 1× | ~2× |

When in doubt, leave `full_state_update=True`. Override only when state is summable/concatable.

---

## 10. Wrapper picking guide (one-liner each)

```text
BootStrapper       → confidence intervals on any metric
MetricTracker      → "best so far" across epochs
Running            → rolling-window metric over last N updates
MinMaxMetric       → track min/max of any scalar metric
MultioutputWrapper → same metric, K parallel outputs
MultitaskWrapper   → different metrics, K different tasks
ClasswiseWrapper   → per-class breakdown of a per-class metric
FeatureShare       → share encoder features across multimodal metrics
Transformations    → preprocess inputs before update
```

---

## 11. NDCG formula card

```
DCG  @k = Σ_{i=1..k}  (2^rel_i − 1) / log2(i + 1)
IDCG @k = DCG on the IDEAL ordering of relevances
NDCG @k = DCG @ k / IDCG @ k    ∈ [0, 1]
```

For binary relevance (`target ∈ {0,1}`), the gain `2^rel − 1` collapses to just `rel`, and NDCG @ k becomes the position-discounted average of relevant hits in top-k.

---

## 12. F1 / precision / recall formula card

```
P = TP / (TP + FP)        "Of what we predicted positive, how many were right?"
R = TP / (TP + FN)        "Of all true positives, how many did we catch?"
F1 = 2·P·R / (P + R)      Harmonic mean of P and R
Fβ = (1+β²)·P·R / (β²·P + R)   β > 1 weights R more; β < 1 weights P more
```

**Quick numerical anchors**:

```
P=0.5, R=0.5   → F1 = 0.500
P=0.6, R=0.4   → F1 = 0.480
P=0.9, R=0.1   → F1 = 0.180   ← imbalance hurts F1
P=1.0, R=1.0   → F1 = 1.000
P=0,   R=0     → F1 = 0       (with zero_division=0)
```

---

## 13. FID formula card

```
FID(real, fake) =  ||μ_r − μ_f||²  +  Tr( Σ_r + Σ_f − 2·sqrt(Σ_r · Σ_f) )

where (μ, Σ) are the mean and covariance of Inception V3 pool3 activations
on the respective image set.
```

**KID** uses an MMD estimator over the same activations — unbiased at small N (FID isn't).

---

## 14. The metric-to-business-KPI bridges

```text
+--------------------+         +------------------------+
|  ML number         |  ─→    |  Business KPI           |
+--------------------+         +------------------------+
| AUROC, F1, NDCG    | (1) Cost-of-error matrix         |
| (model quality)    |     map TP/FP to dollars         |
|                    |                                   |
|                    | (2) Top-k truncation              |
|                    |     business shows top K          |
|                    |                                   |
|                    | (3) Threshold operating point     |
|                    |     hit a precision/recall floor  |
|                    |                                   |
|                    | (4) Calibration                   |
|                    |     P used directly in pricing    |
+--------------------+         +------------------------+
```

Always answer "what metric?" at all three layers:
1. **Model quality** — TorchMetrics number.
2. **Decision quality** — operating-point / dollar-loss.
3. **Outcome** — A/B-tested business KPI.

---

## 15. Lightning logging cheat sheet

```python
# RIGHT — pass the metric instance
self.log("acc", self.train_acc, on_step=True, on_epoch=True)
self.log_dict(self.train_metrics(logits, y), on_step=True, on_epoch=True)

# WRONG — eagerly evaluates and breaks lifecycle
self.log("acc", self.train_acc.compute())

# RIGHT for validation — explicit lifecycle
def validation_step(self, batch, _):
    self.val_metrics.update(self(batch.x), batch.y)
def on_validation_epoch_end(self):
    self.log_dict(self.val_metrics.compute())
    self.val_metrics.reset()
```

---

## 16. Common error → fix table

| Error message | Root cause | Fix |
|---|---|---|
| Expected all tensors to be on the same device | Metric on CPU, inputs on GPU | `metric.to(device)` |
| The Metric has already been synced | Manual `sync()` without `unsync()` | Pair them or use the context |
| RuntimeError: One of the differentiated Tensors does not require grad | Non-differentiable metric in loss | Use a differentiable surrogate |
| compute() returns NaN per-class | Class with zero support | Set `zero_division=` or document NaN |
| OOM during validation | List-state metric on huge eval | `compute_on_cpu=True` |
| Different value across runs | Sampler not seeded | `DistributedSampler(..., seed=...)` + `set_epoch` |
| compute() warning "called before update" | Empty epoch / exception in update | Check `_update_count` |

---

## 17. The "30-second metric pitch" template

> *"This model is **[metric] = X**. We chose [metric] because **[reason — imbalance / cost asymmetry / ranking / calibration]**. The business cost of being wrong here is **[FN cost vs FP cost]**, which is why we're at the **[operating-point]** rather than the textbook default. Confidence interval is **±Y** from a 1000-sample paired bootstrap."*

Memorize the structure. Fill in the blanks per scenario.

---

## 18. The "5 numbers you should always know"

For any classification model in production, you need these five numbers, refreshed daily:

1. **Headline metric** — F1, AUROC, NDCG (whatever's primary).
2. **Worst segment** — minimum metric across (region × device × version) buckets.
3. **Calibration error** — ECE on a fixed bin scheme.
4. **Coverage** — fraction of inputs the model returns a confident answer for.
5. **Drift signal** — KL divergence vs. a frozen reference distribution.

Together, these five tell you "is the model still doing its job?" Print them on a sticky note.
