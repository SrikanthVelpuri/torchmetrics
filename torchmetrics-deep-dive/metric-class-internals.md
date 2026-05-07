---
title: Metric Class Internals
nav_order: 4
---

# Metric Class Internals

A line-by-line tour of `src/torchmetrics/metric.py`. Read this once and you will be able to read any metric in the library, write your own, and debug DDP problems with confidence.

---

## The class signature

```python
class Metric(Module, ABC):
    is_differentiable:  Optional[bool] = None
    higher_is_better:   Optional[bool] = None
    full_state_update:  Optional[bool] = None
    plot_lower_bound:   Optional[float] = None
    plot_upper_bound:   Optional[float] = None
    plot_legend_name:   Optional[str]   = None
```

These class attributes are **declarative metadata**. Subclasses set them so other parts of the system (Lightning, plotting, monitoring) can introspect:

- `is_differentiable=True` allows the metric to be used inside a loss.
- `higher_is_better` controls "best so far" tracking in `MinMaxMetric` / `MetricTracker`.
- `full_state_update` — see [Core Concepts]({{ "./core-concepts.md" | relative_url }}); decides which `forward()` path is used.
- The `plot_*` attributes seed the default y-axis range when calling `metric.plot()`.

`Metric` extends both `nn.Module` (so `.to()`, `state_dict()`, hooks, JIT all work) and `ABC` (so `update` and `compute` must be implemented).

---

## `__init__` — what gets set up

The interesting parts of `__init__` (paraphrased):

```python
def __init__(self, **kwargs):
    super().__init__()
    torch._C._log_api_usage_once(f"torchmetrics.metric.{type(self).__name__}")  # telemetry hook for PyTorch

    # device + dtype tracking (used by .to() and add_state)
    self._device = torch.get_default_device() if _TORCH_GREATER_EQUAL_2_3 else torch.empty(0).device
    self._dtype  = torch.get_default_dtype()

    # behavior flags pulled out of kwargs (each validated)
    self.compute_on_cpu       = kwargs.pop("compute_on_cpu", False)
    self.dist_sync_on_step    = kwargs.pop("dist_sync_on_step", False)
    self.process_group        = kwargs.pop("process_group", None)
    self.dist_sync_fn         = kwargs.pop("dist_sync_fn", None)
    self.distributed_available_fn = kwargs.pop("distributed_available_fn", None) or jit_distributed_available
    self.sync_on_compute      = kwargs.pop("sync_on_compute", True)
    self.compute_with_cache   = kwargs.pop("compute_with_cache", True)

    if kwargs:
        raise ValueError(f"Unexpected keyword arguments: {sorted(kwargs)}")

    # Wrap user methods so the base class can intercept them
    self._update_signature = inspect.signature(self.update)
    self.update  = self._wrap_update(self.update)
    self.compute = self._wrap_compute(self.compute)

    # bookkeeping
    self._computed       = None        # cache for compute()
    self._forward_cache  = None        # last forward() return
    self._update_count   = 0
    self._to_sync        = self.sync_on_compute
    self._should_unsync  = True
    self._enable_grad    = False
    self._dtype_convert  = False

    # state machinery
    self._defaults:    dict[str, Union[list, Tensor]] = {}
    self._persistent:  dict[str, bool] = {}
    self._reductions:  dict[str, Union[str, Callable, None]] = {}

    self._is_synced = False
    self._cache:     Optional[dict[str, Union[list[Tensor], Tensor]]] = None
```

Three things are worth pausing on.

### 1. `update` and `compute` are wrapped

The user's `update` is replaced (`self.update = self._wrap_update(self.update)`) with a wrapper that:

- Bumps `self._update_count`.
- Clears `self._computed` (the cache).
- Sets `torch.set_grad_enabled(self._enable_grad)`.
- Catches the "Expected all tensors to be on the same device" error and rewrites it with a friendly hint pointing the user to `.to(device)`.
- Optionally moves list states to CPU after the update (`compute_on_cpu`).

The wrap on `compute` is even more interesting — covered below.

### 2. Three parallel dicts hold all metric state

```python
self._defaults    # name -> initial tensor / list (for reset)
self._persistent  # name -> bool (does it go in state_dict?)
self._reductions  # name -> reduction function for DDP all_gather
```

`add_state(name, default, dist_reduce_fx, persistent)` populates all three. The actual current value of the state lives as a normal attribute `self.<name>`.

### 3. Telemetry hook

`torch._C._log_api_usage_once(...)` is the same telemetry hook PyTorch core uses internally — calling it lets PyTorch tally how often `Metric` subclasses are constructed (anonymous, opt-out). Worth knowing if you're auditing for telemetry.

---

## `add_state` — the most important user-facing method

```python
metric.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
metric.add_state("preds", default=[],            dist_reduce_fx="cat")
```

Internally:

1. Validates that `default` is either a tensor or an empty list.
2. Maps the string `dist_reduce_fx` to a real callable (`dim_zero_sum`, `dim_zero_mean`, `dim_zero_min`, `dim_zero_max`, `dim_zero_cat`).
3. If `default` is a tensor, makes it `.contiguous()` (matters for `all_gather`).
4. `setattr(self, name, default)` — exposes the state as a normal attribute.
5. Stores a deep copy in `self._defaults`, plus the persistence flag and reduction.

**List state caveat (from the docstring):** when `reset()` is called, list states are emptied — so any reference you held before `reset()` may now be a dangling list. If you need to keep the values, copy them first.

---

## `forward()` — the dual-purpose entry point

`forward(*args, **kwargs)` is what runs when you call the metric like a function: `value = metric(preds, target)`.

```python
def forward(self, *args, **kwargs):
    if self._is_synced:
        raise TorchMetricsUserError("can't forward while synced")

    if self.full_state_update or self.full_state_update is None or self.dist_sync_on_step:
        self._forward_cache = self._forward_full_state_update(*args, **kwargs)
    else:
        self._forward_cache = self._forward_reduce_state_update(*args, **kwargs)
    return self._forward_cache
```

Two implementations, picked at runtime.

### `_forward_full_state_update` — slow & safe

Two `update()` calls per batch:

```text
update(batch)                # accumulate global
save_state                   # save running totals
reset(); update(batch)       # only this batch
batch_value = compute()      # value for this batch only
restore_state                # put running totals back
```

This works for *any* metric — even ones whose `update` reads the previous state.

### `_forward_reduce_state_update` — fast path

Only one `update()` per batch:

```text
save_state; reset()
update(batch)                # only this batch
batch_value = compute()      # value for this batch
merge saved_state with batch_state via dist_reduce_fx
```

The "merge" step calls `_reduce_states`, which knows about `dim_zero_sum/mean/min/max/cat` — so as long as your state reduction is a known one (or a callable that you supply), this works correctly.

---

## `_sync_dist` — distributed magic

When `compute()` is called, it (transparently) wraps the body in a `sync()` / `unsync()` context that, if running under DDP, calls `_sync_dist`:

```python
def _sync_dist(self, dist_sync_fn=gather_all_tensors, process_group=None):
    input_dict = {attr: getattr(self, attr) for attr in self._reductions}

    # pre-concat list states locally, to reduce all_gather count
    for attr, fn in self._reductions.items():
        if fn == dim_zero_cat and isinstance(input_dict[attr], list) and len(input_dict[attr]) > 1:
            input_dict[attr] = [dim_zero_cat(input_dict[attr])]
        if _TORCH_GREATER_EQUAL_2_1 and fn == dim_zero_cat \
           and isinstance(input_dict[attr], list) and len(input_dict[attr]) == 0:
            # corner case: this rank received nothing — emit empty tensor so all_gather still works
            input_dict[attr] = [torch.tensor([], device=self.device, dtype=self.dtype)]

    output = apply_to_collection(input_dict, Tensor, dist_sync_fn,
                                 group=process_group or self.process_group)

    for attr, fn in self._reductions.items():
        if isinstance(output[attr], list) and len(output[attr]) == 0:
            setattr(self, attr, []); continue
        if isinstance(output[attr][0], Tensor):
            output[attr] = torch.stack(output[attr])
        elif isinstance(output[attr][0], list):
            output[attr] = _flatten(output[attr])
        reduced = fn(output[attr]) if fn is not None else output[attr]
        setattr(self, attr, reduced)
```

The two subtle details:

1. **List states get pre-concatenated** before the network call. If rank 0 has 12 tensors in `preds`, we cat them locally to one `Tensor` first, so `all_gather` ships *one* payload of shape `[total]` instead of 12 small ones — a big win for many small batches.
2. **Empty-rank handling**: in heavily uneven workloads (e.g. the last few samples don't reach every rank) a rank may have no data. To avoid `all_gather` dimension mismatches, we send an empty tensor of the right device/dtype.

---

## `sync()` / `unsync()` — the user-controllable API

`compute()`'s wrapper does this:

```text
sync(): _is_synced=True; _cache=copy of state; _sync_dist()
result = compute(<synced state>)
unsync(): if _should_unsync: state ← _cache; _is_synced=False
```

That's why calling `compute()` is non-destructive — synced state is restored before returning.

You can also call `metric.sync(...)` and `metric.unsync()` manually when, say, you want to log synced numbers in a custom evaluation loop.

---

## `merge_state` — composing metrics

A surprisingly useful method. `merge_state` lets you fold one metric's state into another:

```python
m1 = SumMetric(); m1.update(1)
m2 = SumMetric(); m2.update(2)
m1.merge_state(m2)
m1.compute()   # tensor(3.)
```

It only works for metrics with `full_state_update=False` (i.e. those whose state is reducible). Custom metrics that violate that have to override `merge_state` themselves.

This is what enables **distributed evaluation across pipelines** — you can run inference on different machines, persist each metric's `metric_state`, and reduce them later offline.

---

## `_wrap_compute` — caching, syncing, error checks

Pseudo-code:

```python
def _wrap_compute(self, compute):
    def wrapped(*args, **kwargs):
        if self._update_count == 0:
            rank_zero_warn("compute() called before update()")
        if self.compute_with_cache and self._computed is not None:
            return self._computed
        with self.sync_context(...):       # all_gather state across ranks
            value = compute(*args, **kwargs)
        if self.compute_with_cache:
            self._computed = value
        return _squeeze_if_scalar(value)
    return wrapped
```

Three behaviors fall out of this wrapper:

- **First-call check** — calling `compute` before any `update` warns loudly.
- **Caching** — repeated `compute()` calls in a row are O(1) until the next `update()`.
- **Sync context** — handled here so individual metrics never need to know about DDP.

---

## State and `state_dict`

By default, metric states are **not** in `state_dict`. They're transient — you don't usually checkpoint them. If you do want to (e.g. mid-epoch resume), pass `persistent=True` to `add_state`:

```python
self.add_state("running_sum", torch.tensor(0.0),
               dist_reduce_fx="sum", persistent=True)
```

Then `model.state_dict()` will include the running sum.

---

## Putting it together: a request-trace through `metric(x, y)`

1. `metric(x, y)` → `nn.Module.__call__` → wrapped `forward`.
2. `forward` chooses `_forward_full_state_update` or `_forward_reduce_state_update`.
3. Either way, `update(x, y)` is invoked. The wrapper bumps `_update_count`, clears `_computed`, sets grad mode, runs your `update`.
4. `compute()` is called for the batch value. Its wrapper:
   - sees `_computed is None` (we just cleared it),
   - if running under DDP and `dist_sync_on_step=True`, `all_gather`s state,
   - runs your `compute`,
   - returns the value.
5. The forward path restores global state (full path) or merges (reduce path).
6. Result is stored in `self._forward_cache` and returned.

---

## Reading list

- `src/torchmetrics/metric.py` — the file we just dissected.
- `src/torchmetrics/utilities/distributed.py` — `gather_all_tensors`.
- `src/torchmetrics/utilities/data.py` — `dim_zero_*` reductions.
- `src/torchmetrics/aggregation.py` — the simplest concrete metrics; great to read alongside `metric.py`.
