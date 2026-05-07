---
title: Testing & Validation
nav_order: 13
---

# Testing & Validation

How TorchMetrics tests itself, and how you should test the metrics you write.

---

## How TorchMetrics tests itself

The `tests/` directory mirrors `src/torchmetrics/`. For every metric there's a parity test against a *trusted reference* — usually scikit-learn for classification/regression, NumPy/SciPy for statistical metrics, NLTK/sacrebleu for text, MIR-Eval/PESQ for audio, and pycocotools for detection.

The pattern is consistent:

```text
tests/unittests/<domain>/test_<metric>.py
    ├── reference function (calls sklearn etc.)
    ├── inputs fixture (binary / multiclass / multilabel sample tensors)
    ├── class-level test
    │     • DDP and non-DDP
    │     • multiple update calls
    │     • single update + forward
    │     • half-precision
    │     • torch.compile (recent versions)
    │     • differentiability if applicable
    │     • plot smoke test
    └── functional-level test
          • parity vs. reference
          • numerical stability (zeros, NaNs, huge inputs)
```

A typical assertion:

```python
def test_accuracy_class():
    metric = MulticlassAccuracy(num_classes=3, average="macro")
    metric.update(preds, target)
    out = metric.compute()
    ref = sklearn.metrics.balanced_accuracy_score(...)
    assert torch.allclose(out, torch.tensor(ref), atol=1e-6)
```

What's worth absorbing:

1. **Two parallel test suites** — class-level and functional-level — because both APIs must match.
2. **DDP tests** — using `pytest_ddp` infrastructure that spawns multiple processes. This is the only honest way to validate `_sync_dist` behavior.
3. **Edge cases are first-class** — every metric has tests for "all positives," "all negatives," "empty input," "single class present."
4. **Half precision and `torch.compile`** — recent PRs add these as standard test axes.

Tests run on CPU and (in CI) GPU; multi-node tests run on Azure pipelines. The combination is what gives the library its trust budget.

---

## How to test your own custom metric

Treat your custom metric as a black box and write four kinds of tests.

### 1. Parity vs. a reference

Pick a trusted reference (sklearn, your hand-written numpy version, a paper's released code) and assert closeness.

```python
import torch, numpy as np
from sklearn.metrics import mean_absolute_error
from my_metrics import WeightedMAE

def test_weighted_mae_parity():
    rng = np.random.default_rng(0)
    n = 1000
    preds   = torch.tensor(rng.normal(size=n).astype(np.float32))
    target  = torch.tensor(rng.normal(size=n).astype(np.float32))
    weights = torch.tensor(rng.uniform(0.1, 2.0, size=n).astype(np.float32))

    m = WeightedMAE()
    m.update(preds, target, weights)
    got = m.compute().item()

    expected = (weights.numpy() * np.abs(preds.numpy() - target.numpy())).sum() \
             / weights.numpy().sum()

    assert abs(got - expected) < 1e-5
```

### 2. Lifecycle — multiple updates equal one update

This is the single test that catches almost all "I forgot to accumulate" bugs.

```python
def test_multiple_updates_equal_single():
    m1 = WeightedMAE(); m1.update(preds, target, weights)
    m2 = WeightedMAE()
    half = preds.shape[0] // 2
    m2.update(preds[:half], target[:half], weights[:half])
    m2.update(preds[half:], target[half:], weights[half:])
    assert torch.allclose(m1.compute(), m2.compute(), atol=1e-6)
```

If this fails, your `update` is mutating in a non-summable way.

### 3. Reset

```python
def test_reset_returns_to_default():
    m = WeightedMAE()
    m.update(preds, target, weights)
    state_before = {k: v.clone() if torch.is_tensor(v) else list(v) for k, v in m.metric_state.items()}
    m.reset()
    for name, default in m._defaults.items():
        cur = getattr(m, name)
        if torch.is_tensor(default):
            assert torch.equal(cur, default)
        else:
            assert cur == default
```

### 4. DDP

Use the same `pytest_ddp` pattern TorchMetrics uses. Minimal sketch:

```python
def _ddp_worker(rank, world_size, port):
    init_distributed(rank, world_size, port)
    m = WeightedMAE().to(rank)
    # each rank sees a different slice
    slice_ = (slice(rank, None, world_size))
    m.update(preds[slice_], target[slice_], weights[slice_])
    out = m.compute()
    if rank == 0:
        # compare against single-process result
        m_single = WeightedMAE(); m_single.update(preds, target, weights)
        assert torch.allclose(out, m_single.compute(), atol=1e-5)

def test_ddp_correctness():
    spawn(_ddp_worker, 4)
```

If a custom metric passes parity + lifecycle + reset + DDP, it is production-ready.

---

## A self-test before merging a PR

The library's own contribution checklist is a good blueprint:

1. [ ] Unit test against a trusted reference, multiple input regimes (perfect, random, adversarial).
2. [ ] Lifecycle test (multi-update equivalence, reset).
3. [ ] DDP test if the metric has any state.
4. [ ] Half-precision smoke test.
5. [ ] Doctest in the docstring.
6. [ ] `is_differentiable` flag is correct (or explicit `False`).
7. [ ] `higher_is_better` is correct, especially for trackers.
8. [ ] `full_state_update` is set deliberately.
9. [ ] An RST page in `docs/source/<domain>/`.
10. [ ] A line in the changelog.

---

## Common test mistakes

- **Comparing raw float tensors with `==`**. Use `torch.allclose(..., atol=...)`.
- **Comparing means computed in float32 vs. float64**. Some references (sklearn) do float64 internally.
- **Tests that pass on the same machine but fail in CI**. Usually an unset random seed or device-default difference. Pin both.
- **DDP tests that pass with `world_size=1`**. The bug only appears at ≥ 2 ranks. Always test ≥ 2.
- **Forgetting that list states are emptied on `reset()`**. Don't hold references to them.

---

## Continuous benchmarking

For metrics that have a "happy" runtime budget (e.g. AUROC on 1M samples in < 1s), TorchMetrics maintains lightweight benchmarks. If you write a metric that's part of a hot eval loop, add a perf test that asserts the runtime isn't catastrophically regressed by a refactor.
