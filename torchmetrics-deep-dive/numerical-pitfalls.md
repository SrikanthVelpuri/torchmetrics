---
title: Numerical Pitfalls
nav_order: 28
---

# Numerical Pitfalls — The Senior-Interview Layer

Above mid-level, interviewers stop asking "what is F1" and start asking "your F1 differs from your colleague's F1 by 1e-4 — why?" This page is the answers to those.

> **Mnemonic**: **NIPS** — NaN, Infinity, Precision, Scaling.

---

## 1. NaN propagation — the single bad sample that destroys everything

A single NaN in a sum-reduced state poisons the whole running total.

```python
running = torch.tensor(0.0)
running += torch.tensor([1.0, 2.0, float("nan"), 4.0]).sum()
# running is now NaN forever, regardless of subsequent updates
```

**How it happens in TorchMetrics**:

- Numerical instability in your model (one batch outputs Inf logits → softmax → NaN).
- A bad row in your eval set (Inf target, NaN preprocessing).
- Division by zero inside `update()` (rare but possible in custom metrics).

**Defenses**:

1. **Filter at the boundary**, not inside `update()`. Drop NaN rows before calling `metric.update(...)`.
2. **NaN-strategy params** — many TorchMetrics metrics accept `nan_strategy="error" | "warn" | "ignore" | "replace"`. Read the docstring; pick deliberately.
3. **Track NaN rate** as its own metric. Custom `MeanMetric()` updated with `(value.isnan().sum(), total)` is a one-line drift sensor.
4. **Audit logs**. If your headline metric goes NaN in production, you want to know which sample caused it.

---

## 2. Infinity and `log(0)`

Calibration metrics, log-loss, perplexity, and anything in log-space hit this.

```python
# Naive log-loss
log_p = torch.log(probs)         # if any prob == 0, log_p = -Inf
loss = -((target * log_p).sum())  # Inf in the running sum
```

**Defenses**:

```python
# Numerically stable
log_p = torch.log(probs.clamp(min=1e-12))
# OR use log_softmax directly on logits
log_p = F.log_softmax(logits, dim=-1)
```

**TorchMetrics specifics**:

- `Perplexity` already uses `log_softmax` internally — safe.
- Custom calibration metrics need explicit `clamp(min=eps)` before `log`.
- BLEU's geometric mean of n-gram precisions: if any precision is 0, the result is 0 (not Inf, but a weird discontinuity). Smoothing methods exist (`add-1`, Chen-Cherry).

---

## 3. Precision — float16 / bfloat16 underflow

Mixed-precision training is everywhere now. Metrics often live in fp32 or fp64 internally, but if you accidentally cast state down:

```python
self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
# Then somewhere:
self.sum = self.sum.half()    # NEVER do this
```

You're now accumulating in fp16. After ~65k updates of order ~1.0, the running sum saturates because fp16 max ≈ 65504.

**Defenses**:

1. **Keep accumulators in fp32 or fp64**, even when inputs are fp16/bf16. TorchMetrics does this by default.
2. **Cast inputs explicitly**: `self.sum += x.float().sum()` if you suspect inputs are reduced precision.
3. **Use Kahan summation** for very long-running aggregations:

```python
# Kahan summation reduces float-rounding error
class KahanSum:
    def __init__(self):
        self.sum = 0.0
        self.c   = 0.0   # compensation
    def add(self, x):
        y = x - self.c
        t = self.sum + y
        self.c = (t - self.sum) - y
        self.sum = t
```

For most TorchMetrics use cases, fp32 accumulation is enough. Kahan only matters at billions-of-updates scale.

---

## 4. Float-reduction non-associativity

```
(a + b) + c  ≠  a + (b + c)    at machine precision
```

This means **DDP all_gather order can change the result by 1e-7 ish**.

**When it matters**:

- Reproducibility tests across runs.
- Comparing your number to sklearn / numpy reference.
- Cross-rank consistency checks.

**Defenses**:

- For deterministic reductions, set `torch.use_deterministic_algorithms(True)` and `NCCL_ALGO=Tree` (slower but order-stable).
- Accept that bit-exact comparison fails; use `torch.allclose(atol=1e-6)`.
- If you genuinely need bit-exact, run on a single rank (defeats the purpose of DDP).

---

## 5. Catastrophic cancellation

```python
# Variance — naive
mean = x.mean()
var  = ((x - mean) ** 2).mean()           # OK

# Variance — naive streaming (BAD)
sum_sq = (x ** 2).sum()
sum_   = x.sum()
n      = x.numel()
var    = sum_sq / n - (sum_ / n) ** 2     # Catastrophic cancellation when var ≪ mean²
```

`PearsonCorrCoef` uses Welford's algorithm internally specifically to avoid this. If you write a custom metric that computes variance, **use Welford's** or just use TorchMetrics' aggregation primitives.

---

## 6. Edge cases — empty input, single sample, single class

```python
# What does this print?
metric = F1Score(task="multiclass", num_classes=10)
metric.update(torch.empty(0, 10), torch.empty(0, dtype=torch.long))
print(metric.compute())   # ?
```

The right behavior depends on your `zero_division` policy. Different metrics handle empty input differently:

- `MeanMetric()` on empty: returns NaN.
- `Accuracy()` on empty: returns NaN or warning depending on version.
- `AUROC()` on empty: raises (you can't compute AUC on no data).
- `BLEUScore()` on empty: returns 0 (with warning).

**Test your custom metrics on empty input.** If it returns NaN or raises, document it. If it silently returns 0, that's almost always a bug.

---

## 7. Single-class eval set

If your eval set has only positive samples (or only negative), AUROC, AP, ROC, PR are all undefined.

```python
preds  = torch.tensor([0.1, 0.2, 0.3, 0.4])
target = torch.tensor([1, 1, 1, 1])              # all positive
auroc.update(preds, target)
auroc.compute()    # raises or warns
```

**Defenses**:

1. Catch this at eval-set construction time. Assert ≥ 1 positive AND ≥ 1 negative.
2. For multi-class, ensure each class has ≥ 1 sample if you're computing per-class metrics.
3. `MultilabelF1Score` handles per-label empty support via `zero_division` parameter.

---

## 8. Tied scores in AUROC

```python
preds  = torch.tensor([0.5, 0.5, 0.5, 0.5])
target = torch.tensor([1,   0,   1,   0])
# AUROC = 0.5 (random order tie-breaking)
```

Tie-breaking conventions differ across libraries:

- **sklearn**: averages the AUC over all possible tie-break orderings (correct theory).
- **TorchMetrics**: matches sklearn's step interpolation for AP. For AUROC, `roc_auc_score` parity is enforced.
- **Some other libraries**: just sort and break ties however the sort algorithm does — non-deterministic.

If your AUROC differs from a colleague's by 1e-3, suspect tie-breaking.

---

## 9. Bin-edge sensitivity in calibration

```python
ece = BinaryCalibrationError(n_bins=15)
```

ECE is sensitive to bin choice. With predictions concentrated near 0 and 1 (modern over-confident nets), equal-width bins put almost everything in two bins. Try:

- **Equal-mass (adaptive) binning** — same number of samples per bin.
- **Different bin counts** — sweep K=5, 10, 15, 20; report all.

Don't trust a single ECE number with default bins for a publication-grade comparison.

---

## 10. Sample-count bias in FID / KID

```python
fid_at_1k  = ...   # bias ≈ +5
fid_at_10k = ...   # bias ≈ +0.5
fid_at_50k = ...   # bias ≈ +0.1
```

FID's covariance estimator is biased downward at small N — you systematically *underestimate* the real distance. Two papers reporting "FID = 8" at different N are not comparable.

**Defenses**:

- Always report N alongside FID.
- Use KID for small-sample regimes (unbiased estimator).
- Pin sample count across experiments.

---

## 11. PCC / Spearman streaming numerical stability

`PearsonCorrCoef` uses Welford's algorithm — numerically stable streaming Pearson. Spearman *can't* be made streaming exactly because rank requires the full population.

If you implement custom Pearson, the naive formula:

```python
r = (Σxy - n·x̄·ȳ) / sqrt((Σx² - n·x̄²)(Σy² - n·ȳ²))
```

…suffers catastrophic cancellation when correlation is near 1. The Welford / co-moment update is the right approach.

---

## 12. Wrong-device subtleties

```python
metric.to("cuda:0")
# ... later, after some refactor:
metric.cpu()
metric.update(preds_on_cuda, target_on_cuda)
# device-mismatch error
```

If the metric is on CPU but you forget and pass GPU inputs, the error is loud (TorchMetrics rewrites it with a hint). The silent version:

```python
metric = MyMetric()              # CPU
metric.add_state("buffer", torch.tensor(0.0), dist_reduce_fx="sum")
# ... metric.to("cuda")
# but if MyMetric.update creates intermediate tensors with torch.zeros(...),
# they default to CPU unless you explicitly say device=self.device
```

**Defense in custom metrics**:

```python
def update(self, x, y):
    intermediate = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
    # ...
```

Always inherit device from the input, not from the global default.

---

## 13. Half-precision metric input

Modern training is bf16. If your `update()` sees bf16 inputs:

```python
def update(self, preds, target):
    diff = preds - target              # bf16
    self.sum += diff.pow(2).sum()      # bf16 sum
```

Accumulation in bf16 saturates around N=2^7. For an MSE accumulator over a 1M-sample eval, you'll lose enormous precision.

**Fix**: cast to fp32 before accumulating:

```python
self.sum += diff.float().pow(2).sum()
```

Many TorchMetrics built-ins do this internally.

---

## 14. The `torch.compile` and dynamic-shape pitfall

`torch.compile` doesn't always play well with metrics that have dynamic-shape list states. The first compute() after `torch.compile` may recompile every epoch.

**Defenses**:

- Don't `torch.compile` the metric itself; compile the model only.
- For tensor-state metrics, compile is generally fine.
- For list-state, prefer `compute_on_cpu=True` with no compile.

---

## 15. Reset doesn't truncate references

```python
metric = AUROC(task="binary")
metric.update(preds, target)
my_preds = metric.preds         # save reference
metric.reset()
print(my_preds)                 # ← what's in here?
```

`reset()` replaces `metric.preds` with a fresh empty list. The old list (and its tensors) live on through `my_preds` until you drop the reference. This is by design — but if you held a reference to a giant list state, it doesn't get GC'd until you drop it.

---

## Cheat sheet — symptoms → cause → fix

| Symptom | Probable cause | Fix |
|---|---|---|
| Metric goes NaN suddenly | NaN in input | Filter; track NaN rate as separate metric |
| Metric value flatlines | fp16 saturation | Cast accumulators to fp32 |
| Bit-exact reproducibility fails | Float non-associativity | Use `allclose`; deterministic algorithms |
| AUROC differs from sklearn by 1e-3 | Tie-breaking | Document the convention |
| ECE jumps with bin count | Bin sensitivity | Adaptive bins; sweep K |
| FID smaller at small N | Sample-bias | Always report N; pair with KID |
| Variance is negative | Catastrophic cancellation | Welford's algorithm |
| Empty eval breaks compute | Single-class data | Assert at eval-set construction |
| Custom Pearson noisy near r=1 | Naive formula | Welford / co-moment |
| Lightning logs garbage | Metric on wrong device | Register as module attr |

Memorize the left column. The right column is the search query.
