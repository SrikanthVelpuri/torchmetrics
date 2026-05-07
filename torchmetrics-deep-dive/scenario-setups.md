---
title: Scenario Setups
nav_order: 19
---

# Scenario Setups — How Metrics Are Wired in Real ML Systems

This page is a recipe book. For each common ML scenario, it shows **the exact metric configuration** you'd use in production: which metrics, which wrappers, where they live in the training/serving stack, and how they're logged.

Each section ends with the multi-level interview drill-down you should expect.

---

## Scenario 1 — Image classification training (ImageNet-style)

**Stack.** PyTorch + Lightning, multi-GPU DDP, mixed precision, ~1M images.

```python
import torch, pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, MulticlassCalibrationError,
)
from torchmetrics.wrappers import ClasswiseWrapper

class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes=1000, class_names=None):
        super().__init__()
        self.model = build_resnet50(num_classes)

        base = MetricCollection({
            "top1": MulticlassAccuracy(num_classes=num_classes, top_k=1),
            "top5": MulticlassAccuracy(num_classes=num_classes, top_k=5),
            "f1":   MulticlassF1Score(num_classes=num_classes, average="macro"),
            "ece":  MulticlassCalibrationError(num_classes=num_classes, n_bins=15),
        })
        self.train_metrics = base.clone(prefix="train/")
        self.val_metrics   = base.clone(prefix="val/")

        # Per-class diagnostics — only useful at val time, on a subset of metrics
        self.val_per_class = ClasswiseWrapper(
            MulticlassAccuracy(num_classes=num_classes, average=None),
            labels=class_names,
        )

    def training_step(self, batch, _):
        x, y = batch
        logits = self.model(x)
        loss   = torch.nn.functional.cross_entropy(logits, y)
        self.log("train/loss", loss, prog_bar=True)
        self.log_dict(self.train_metrics(logits, y), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self.model(x)
        self.val_metrics.update(logits, y)
        self.val_per_class.update(logits, y)

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute())
        # Don't log all 1000 per-class accuracies as scalars — log worst 10 for diagnosis
        per_class = self.val_per_class.compute()    # dict
        worst = sorted(per_class.items(), key=lambda kv: kv[1])[:10]
        for name, value in worst:
            self.log(f"val/worst_class/{name}", value)
        self.val_metrics.reset(); self.val_per_class.reset()
```

**Why this configuration**

- Top-1 / top-5 are the historical reporting numbers; both kept for benchmark continuity.
- Macro-F1 is reported because long-tail classes are silently ignored by accuracy.
- ECE because downstream consumers (e.g. an ensemble or a calibrated classifier head) may use the probability.
- `ClasswiseWrapper` only on val to avoid bloating train logs.
- "Worst 10 classes" pattern beats logging 1000 per-class series — readable dashboard, still actionable.

#### Interview drill-down

**Q.** Why log both `on_step=True` and `on_epoch=True` for training metrics?

> A. Step series shows training noise / instability mid-epoch (useful when LR-tuning); epoch series gives the clean compare-across-epochs trend.

  **F1.** Doesn't `on_step=True` add DDP barriers?

  > No. Lightning + TorchMetrics does **not** all_gather on `forward()` by default; it only syncs at `compute()` time. So step logging is local-rank, near-zero overhead. A small inter-rank divergence in step metrics is expected and harmless.

    **F1.1.** What if I want a step-level *globally synced* metric for debugging instability?

    > Set `dist_sync_on_step=True` on the metric. **Costs you a global barrier per step.** Acceptable for short debug runs; unacceptable for prod training.

      **F1.1.1.** How would you avoid the barrier but still get a quick global view?

      > Use Lightning's `sync_dist=True` on a *scalar loss* — it syncs the reduced number, not the metric state. Different mechanism, much cheaper, but only valid for sums/means.

---

## Scenario 2 — Object detection (COCO mAP)

**Stack.** Single-machine multi-GPU, ~120k images.

```python
from torchmetrics.detection import MeanAveragePrecision

map_metric = MeanAveragePrecision(
    box_format="xyxy",
    iou_type="bbox",
    iou_thresholds=None,           # default = COCO sweep [0.5, 0.55, ..., 0.95]
    rec_thresholds=None,           # default = 101-point recall
    max_detection_thresholds=[1, 10, 100],
    class_metrics=True,            # adds per-class AP at compute time
    compute_on_cpu=True,           # critical: COCO eval is RAM-heavy
)
```

**Inside the `validation_step`**

```python
def validation_step(self, batch, _):
    images, targets = batch
    preds = self.model(images)        # list[dict] with boxes/scores/labels
    self.map_metric.update(preds, targets)

def on_validation_epoch_end(self):
    out = self.map_metric.compute()
    # mAP, mAP_50, mAP_75, mAP_small/medium/large
    self.log_dict({f"val/{k}": v for k, v in out.items() if v.numel() == 1})
    # per-class is a tensor; log min and worst class only
    if "map_per_class" in out:
        self.log("val/map_min_class", out["map_per_class"].min())
    self.map_metric.reset()
```

**Why these knobs**

- `compute_on_cpu=True` → predictions / targets accumulated on host, not GPU. Detection eval easily blows 16 GB GPU.
- `class_metrics=True` → per-class AP for diagnosis, but logged conservatively.
- `max_detection_thresholds` mirrors COCO standard so numbers are comparable to papers.

#### Interview drill-down

**Q.** Why doesn't TorchMetrics give you GPU mAP?

> A. The COCO evaluator (and `pycocotools` / `faster-coco-eval`) runs in NumPy/Cython on CPU. It's not a TorchMetrics limitation — it's that the canonical mAP definition involves sorting + Hungarian-style matching that's awkward to fuse on GPU. TorchMetrics ships the canonical implementation for trust.

  **F1.** What if you need fast eval during training?

  > Use a cheaper proxy: per-class top-1 accuracy on classifier head, plus IoU on box regression head, both implemented as standard tensor-state metrics. Run real mAP only every N validation epochs.

    **F1.1.** Won't proxy metrics drift from real mAP?

    > Yes; track the correlation between proxy and full mAP on a subset; recalibrate the proxy if correlation drops below 0.9.

      **F1.1.1.** Why 0.9 specifically?

      > Empirical: at correlation < 0.9, the proxy starts choosing different models than full mAP would. Below that you risk shipping a worse model. At ≥ 0.95 you can trust it as a gating metric.

---

## Scenario 3 — Semantic segmentation (medical imaging)

**Stack.** 3D volumes, single-GPU per patient, custom loss.

```python
from torchmetrics.segmentation import (
    GeneralizedDiceScore, MeanIoU, HausdorffDistance,
)

seg_metrics = MetricCollection({
    "dice":      GeneralizedDiceScore(num_classes=NUM_CLASSES, weight_type="square"),
    "iou":       MeanIoU(num_classes=NUM_CLASSES),
    "hausdorff": HausdorffDistance(num_classes=NUM_CLASSES, distance_metric="euclidean"),
})
```

**Why this stack**

- Dice is the standard medical-imaging metric; weighting by `square` gives larger weight to more frequent classes (typical for tumor vs. background imbalance).
- IoU as a sanity-check (Dice and IoU are monotonic but not identical).
- Hausdorff for boundary quality — *the* metric reviewers ask about for tumor segmentation.

#### Interview drill-down

**Q.** Why both Dice and IoU?

> A. They're monotonic but not affine. Dice rewards near-perfect overlap more steeply; IoU is closer to a "match fraction." Reviewers expect both.

  **F1.** Why is Hausdorff distance brittle?

  > It's the *worst-case* boundary error. A single rogue voxel dominates. For clinical reporting, prefer **average symmetric surface distance (ASSD)** or **95th-percentile Hausdorff (HD95)** which clip the tail.

    **F1.1.** TorchMetrics doesn't ship HD95 directly. How would you add it?

    > Subclass `HausdorffDistance`. Override `compute()` to take the 95th percentile of the per-voxel distance distribution rather than the max. List state of pairwise distances, `cat` reduction, percentile in compute.

      **F1.1.1.** Why list state and not running quantile estimator?

      > Per-volume eval is small (one volume → one distance distribution). The accumulator is across volumes only. List state is fine.

---

## Scenario 4 — Generative image model (text-to-image, FID + CLIP)

**Stack.** Multi-GPU eval, ~10k generated samples vs. 50k real.

```python
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

gen_metrics = MetricCollection({
    "fid":  FrechetInceptionDistance(feature=2048, normalize=True),
    "kid":  KernelInceptionDistance(feature=2048, subset_size=1000),
    "lpips": LearnedPerceptualImagePatchSimilarity(net_type="vgg"),
}).to(device)

clip = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
```

**Eval pattern**

```python
# 1. Pre-feed real images once and cache the FID/KID stats
for batch in real_loader:
    gen_metrics["fid"].update(batch, real=True)
    gen_metrics["kid"].update(batch, real=True)

# 2. Stream generated images
for prompt, generated in eval_pairs:
    gen_metrics["fid"].update(generated, real=False)
    gen_metrics["kid"].update(generated, real=False)
    clip.update(generated, prompt)

print(gen_metrics.compute(), clip.compute())
```

**Why this configuration**

- FID is the canonical fidelity number; KID is the small-sample-friendly alternative — track both, don't trust either alone.
- LPIPS for paired comparisons (e.g. style-transfer, inpainting where you have a reference).
- CLIPScore for *prompt fidelity*, the dimension FID is blind to.

#### Interview drill-down

**Q.** Why pre-feed real images instead of recomputing them every eval?

> A. FID's "real" statistics depend only on the real distribution and the feature extractor. Recomputing them per eval wastes 50k forward passes through Inception. Cache once.

  **F1.** How do you cache them practically?

  > Persist `metric.metric_state` to disk after the real-feed loop. On the next run, `metric.merge_state(loaded_state)` restores the running mean/cov accumulators. Both states are tensor-state with `sum` reduction → DDP-friendly.

    **F1.1.** What invalidates the cache?

    > Anything that changes the real-image preprocessing or the feature extractor: image resolution, normalization, Inception V3 weights, dtype. Hash the (preprocessing config + extractor checksum) into the cache filename.

      **F1.1.1.** A teammate gets different FID numbers on the "same" data. What do you check first?

      > 1. Inception backbone version. 2. Preprocessing pipeline (resize order, interpolation mode). 3. Image data range (0-1 vs 0-255). 4. Sample count (FID is biased at small N). 5. TorchMetrics version.

---

## Scenario 5 — LLM training / fine-tuning eval

**Stack.** Multi-node, FSDP, periodic eval on suite of benchmarks (MMLU, HumanEval, summarization, RAG QA).

```python
from torchmetrics.text import (
    Perplexity, ROUGEScore, BLEUScore, BERTScore,
)
from torchmetrics import MetricCollection

# General LM eval
lm_metrics = MetricCollection({
    "ppl": Perplexity(ignore_index=-100),
})

# Summarization eval (XSum / CNN-DM)
summ_metrics = MetricCollection({
    "rouge1": ROUGEScore(rouge_keys="rouge1"),
    "rougeL": ROUGEScore(rouge_keys="rougeL"),
    "bertscore": BERTScore(model_name_or_path="microsoft/deberta-xlarge-mnli"),
})

# Code (HumanEval) — pass@k is custom
# MMLU / multi-choice — Accuracy is sufficient
```

**Why these**

- Perplexity is the universal LM training-watcher number; cheap to compute on the held-out shard.
- ROUGE for summarization parity with literature.
- BERTScore for semantic similarity (catches paraphrases ROUGE misses).
- pass@k for code (custom metric; canonical implementation involves running candidate solutions).

#### Interview drill-down

**Q.** Why both ROUGE and BERTScore?

> A. ROUGE is a lexical-overlap metric — quick, deterministic, sensitive to surface form. BERTScore captures semantic similarity but is slower and depends on a reference model. Together they tell you "did the model use the right words" *and* "did it convey the right meaning."

  **F1.** Which one would you optimize directly?

  > Neither. Both are evaluation metrics. The training loss is cross-entropy on next-token. Trying to optimize ROUGE directly through RL (e.g. SCST) gives short-term wins on the metric but degrades human preference.

    **F1.1.** What's the right "ground-truth" eval for an LLM?

    > Human preference (side-by-side comparisons, ELO ratings) is the only honest answer. Automated metrics are proxies — useful for fast iteration, but only the human signal closes the loop.

      **F1.1.1.** What's the role of automated metrics, then?

      > Triage. Automated metrics are cheap enough to run on every commit; human eval is expensive and runs on candidate models. The pipeline is: cross-entropy (every step) → ROUGE/BERTScore/MMLU (every checkpoint) → human SBS (only on shortlist).

**Q.** Perplexity dropped 5 % but ROUGE stayed flat. What do you think happened?

  **F1.** A few hypotheses:

  > (a) Model got better at the easy tail (common phrasings) but didn't change the rare-event tail that drives ROUGE. (b) The eval-set distribution shifted away from the training distribution. (c) The tokenizer changed — perplexity is in the new token space; ROUGE is in word space — so they're not directly comparable across tokenizer versions.

    **F1.1.** How do you isolate (c)?

    > Compute "**bits-per-character**" instead of perplexity — invariant to tokenizer. If bits/char also dropped, the gain is real. If it didn't, the perplexity drop is a tokenizer artifact.

---

## Scenario 6 — Recommender system (offline + online)

**Stack.** Two-tower retrieval + cross-attention reranker. Offline training, online A/B.

**Offline metrics (training)**

```python
from torchmetrics.retrieval import (
    RetrievalRecall, RetrievalNDCG, RetrievalHitRate, RetrievalMRR,
)

retr_metrics = MetricCollection({
    "recall@100": RetrievalRecall(top_k=100),
    "ndcg@10":    RetrievalNDCG(top_k=10),
    "hit@10":     RetrievalHitRate(top_k=10),
    "mrr":        RetrievalMRR(empty_target_action="skip"),
})
```

**Online metrics (production rolling window)**

```python
from torchmetrics.wrappers import Running

rolling_ndcg = Running(RetrievalNDCG(top_k=10), window=50_000).to(device)
```

**Custom business metric**

```python
class IncrementalGMV(Metric):
    higher_is_better = True
    full_state_update = False
    def __init__(self):
        super().__init__()
        self.add_state("test_gmv",     torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("control_gmv",  torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, test_gmv, control_gmv):
        self.test_gmv    += test_gmv.sum()
        self.control_gmv += control_gmv.sum()

    def compute(self):
        return (self.test_gmv - self.control_gmv) / self.control_gmv.clamp(min=1.0)
```

#### Interview drill-down

**Q.** What's the difference between offline NDCG and online business lift?

> A. Offline NDCG measures rank quality on **logged** historical data. The log is biased by the prior recommender (you only saw clicks on what you previously showed). Online lift measures **causal** impact via random exploration or A/B test.

  **F1.** Why do offline gains often fail to materialize online?

  > Distribution shift. Your offline eval assumes the input distribution is the historical one — but a new recommender changes what users see, which changes what they click, which changes the distribution.

    **F1.1.** How do you make offline eval more predictive?

    > (a) Counterfactual evaluation with IPS. (b) Simulation: build a user-response model and replay candidate recommenders against it. (c) Reserve traffic for uniform random exploration to get an unbiased eval set.

      **F1.1.1.** Trade-offs of (c)?

      > Random exploration costs short-term revenue (showing bad recs sometimes). The benefit is unbiased eval, which lets you ship faster and with more confidence. Most large platforms run 1-5 % exploration.

---

## Scenario 7 — Speech recognition (ASR)

**Stack.** Wav2Vec / Whisper-style. 100k+ hours of audio, multi-domain (read speech, conversational, accented).

```python
from torchmetrics.text import (
    WordErrorRate, CharErrorRate, MatchErrorRate, WordInfoLost,
)
from torchmetrics import MetricCollection

asr_metrics = MetricCollection({
    "wer":  WordErrorRate(),
    "cer":  CharErrorRate(),
    "mer":  MatchErrorRate(),
    "wil":  WordInfoLost(),
})

# per-domain breakdown
domain_metrics = {d: asr_metrics.clone(prefix=f"{d}/") for d in ["read", "conv", "accented", "noisy"]}
```

#### Interview drill-down

**Q.** Why CER as well as WER?

> A. WER is undefined when the reference has zero words (rare but real for empty utterances). CER is more granular — useful for languages with no clear word boundaries (Chinese, Japanese) and as a smoother training signal.

  **F1.** Why per-domain WER and not just aggregate?

  > Aggregate WER is dominated by the largest domain. A 5 % global WER can hide 25 % WER on the accented-speech subset, which is the segment that actually matters for fairness.

    **F1.1.** What do you do when accented WER is significantly worse than read-speech?

    > Up-sample / fine-tune on the underrepresented accent; track per-accent WER as an explicit gate in the launch checklist.

      **F1.1.1.** How do you avoid overfitting the accented eval set during fine-tuning?

      > Hold out a *fresh* accent subset (different speakers, recorded in different conditions) only used at the final gate. Never look at it during iteration. The training accent set and the gate accent set must come from disjoint sources.

---

## Scenario 8 — Time-series forecasting (retail / energy / supply chain)

**Stack.** Hierarchical forecasting at SKU × store × day; reconciliation across hierarchy.

```python
from torchmetrics.regression import (
    MeanAbsolutePercentageError,
    SymmetricMeanAbsolutePercentageError,
    WeightedMeanAbsolutePercentageError,
    MeanSquaredError,
)
from torchmetrics import MetricCollection

# Per hierarchy level: SKU, store, region, total
forecasting_metrics = MetricCollection({
    "wmape": WeightedMeanAbsolutePercentageError(),
    "smape": SymmetricMeanAbsolutePercentageError(),
    "rmse":  MeanSquaredError(squared=False),
})
hierarchy_metrics = {
    level: forecasting_metrics.clone(prefix=f"{level}/")
    for level in ["sku", "store", "region", "total"]
}
```

#### Interview drill-down

**Q.** Why wMAPE and not MAPE?

> A. MAPE divides by `y`, exploding when `y → 0` (slow movers, holidays, new SKUs). wMAPE divides total absolute error by total actual — robust on low-volume series, still scale-free.

  **F1.** Why also SMAPE?

  > Different reviewers expect different conventions. Reporting both shields you from "your numbers don't match our team's."

    **F1.1.** What's the canonical metric for a hierarchical forecast?

    > Per-level wMAPE plus a **reconciliation residual** — the inconsistency between bottom-up and top-down forecasts. Custom metric: `mean(|sum_of_children − parent|)`.

      **F1.1.1.** Why does reconciliation residual matter for the business?

      > Inventory + finance teams use *different levels* of the forecast. If they don't add up, the company is allocating inventory to a different total than it's planning revenue against. The residual is a direct measure of cross-functional inconsistency.

---

## Scenario 9 — Anomaly detection (fraud, security, manufacturing)

**Stack.** Highly imbalanced binary classification, often unsupervised at first.

```python
from torchmetrics.classification import (
    BinaryAUROC, BinaryAveragePrecision,
    BinaryRecallAtFixedPrecision,
    BinaryPrecisionAtFixedRecall,
)

anomaly_metrics = MetricCollection({
    "auroc":   BinaryAUROC(),
    "ap":      BinaryAveragePrecision(),
    "rec@p99": BinaryRecallAtFixedPrecision(min_precision=0.99),
    "prec@r80": BinaryPrecisionAtFixedRecall(min_recall=0.80),
})
```

#### Interview drill-down

**Q.** Why AP over AUROC for rare events?

> A. AUROC is dominated by the negative class (TPR sweeps over a denominator inflated by negatives). AP integrates over recall on the positive class. For 0.1 % positive rate, the difference between an "OK" and a "great" model is invisible to AUROC and obvious in AP.

  **F1.** Operating-point: `Recall@FixedPrecision` or `Precision@FixedRecall`?

  > Depends on which constraint is hard. If "we cannot tolerate more than X % false alarms," use `Recall@FixedPrecision`. If "we cannot afford to miss more than Y % of true anomalies," use `Precision@FixedRecall`.

    **F1.1.** What if the data is unsupervised at first (no labels)?

    > Two-stage:
    > 1. Train an unsupervised model (autoencoder, isolation forest).
    > 2. Have humans label a small batch of high-anomaly-score samples.
    > 3. Compute supervised metrics on the labeled subset.
    > 4. Use the labels to tune threshold; retrain a semi-supervised classifier when you have enough labels.

      **F1.1.1.** How do you avoid "label the model's own predictions" bias?

      > Diverse sampling: include uniform-random samples (not just high-score) in the label queue. Importance-weight the labeled subset back to the population when computing metrics.

---

## Scenario 10 — Multi-task / multi-head models

**Stack.** A single model with several heads (e.g. age estimation + emotion + face recognition + landmark detection).

```python
from torchmetrics.wrappers import MultitaskWrapper, MultioutputWrapper
from torchmetrics.regression import MeanAbsoluteError
from torchmetrics.classification import MulticlassF1Score

multi_metrics = MultitaskWrapper({
    "age":      MeanAbsoluteError(),
    "emotion":  MulticlassF1Score(num_classes=7, average="macro"),
    "face_id":  MulticlassF1Score(num_classes=10000, average="macro"),
    "landmark": MultioutputWrapper(MeanAbsoluteError(), num_outputs=68),
})

# Use it
multi_metrics.update(
    {"age": age_pred, "emotion": emo_pred, "face_id": id_pred, "landmark": lm_pred},
    {"age": age_true, "emotion": emo_true, "face_id": id_true, "landmark": lm_true},
)
```

#### Interview drill-down

**Q.** How do you balance per-task metrics into a single launch decision?

> A. You don't, automatically. You set per-task launch gates (each task must not regress more than X %), and a primary task whose lift gates the launch. Aggregate score = a weighted sum is rarely a great idea — a small gain on one task can mask a large regression elsewhere.

  **F1.** What if launches are slowed by the per-task gates conflicting?

  > Pareto reporting. Plot per-task changes; surface only the candidates that Pareto-dominate the production model on the chosen task vector. The decision moves up to a human; the metric layer keeps the picture honest.

    **F1.1.** Where does TorchMetrics' `MultitaskWrapper` end and the human decision begin?

    > `MultitaskWrapper` gives you the per-task numbers (cleanly namespaced, DDP-correct). It does not gate launches — that's a thin layer on top, often a tiny rule engine in your CI / launch tool.

---

## A pattern that holds across all scenarios

**Three layers of metric, always.**

1. A *fast* training-loop metric (single number you watch every step).
2. A *broad* validation-time `MetricCollection` (the dashboard).
3. A *segmented* breakdown (per-class, per-segment, per-domain) for diagnosis.

Every scenario above follows this. If you can articulate which level a metric lives in, you've already won the architecture portion of the interview.
