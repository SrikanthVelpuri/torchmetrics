---
title: Text, Audio, and Image Metrics
nav_order: 8
---

# Text, Audio, and Image Metrics

These domain-specific subpackages live under:

- `torchmetrics.text`
- `torchmetrics.audio`
- `torchmetrics.image`
- `torchmetrics.multimodal`
- `torchmetrics.detection`
- `torchmetrics.segmentation`
- `torchmetrics.video`

Each requires extra dependencies — install via `pip install "torchmetrics[text]"`, `[image]`, `[audio]`, or `[all]`.

---

## Text metrics

### N-gram-based (cheap, deterministic)

| Metric | Class | Use |
|---|---|---|
| BLEU | `BLEUScore` | Machine translation; n-gram precision with brevity penalty. |
| ROUGE | `ROUGEScore` | Summarization; recall-oriented (ROUGE-1/2/L). |
| METEOR | `METEORScore` | MT; aligns synonyms / stems. |
| chrF | `CHRFScore` | Char-n-gram F-score; robust to morphologically-rich languages. |
| TER | `TranslationEditRate` | Edit distance / reference length. |
| Word Error Rate | `WordErrorRate` | ASR — `(S + D + I) / N`. |
| CER | `CharErrorRate` | OCR / ASR at character level. |
| Match Error Rate | `MatchErrorRate` | Variant of WER. |
| Word Info Lost / Preserved | `WordInfoLost`, `WordInfoPreserved` | Information-theoretic ASR scores. |
| Perplexity | `Perplexity` | LM eval on token-level log-probs. |
| ExtendedEditDistance | `ExtendedEditDistance` | Better-aligned variant of edit distance. |

### Embedding-based (semantic, GPU-heavy)

| Metric | Class | Notes |
|---|---|---|
| BERTScore | `BERTScore` | Cosine sim of contextual embeddings (precision/recall/F1). |
| InfoLM | `InfoLM` | Information-theoretic divergence between LM distributions. |

These ship a default model (e.g. `roberta-large`) but you can override `model_name_or_path`. Be careful: the model is loaded **lazily** the first time `update()` is called, on whichever device the metric currently lives on. Move the metric *before* updating.

### Practical text-metric advice

1. **BLEU on a single sentence is meaningless** — it was designed for corpus-level evaluation. Either accumulate over your whole eval set (`update` per sentence, `compute` at the end) or report sentence-BLEU as exactly that, with a footnote.
2. **ROUGE-L is sensitive to tokenization**. Use the same tokenizer as your reference paper.
3. **BERTScore** is GPU-bound. Batch your `update` calls; don't call once per sentence in a Python loop.
4. **Perplexity** wants *token-level log probabilities*, not text. Make sure to align with the same tokenizer your model uses.

---

## Audio metrics

| Metric | Class | What it measures |
|---|---|---|
| SI-SDR | `ScaleInvariantSignalDistortionRatio` | Speech enhancement / source separation. |
| SDR | `SignalDistortionRatio` | Same family, scale-dependent. |
| SNR | `SignalNoiseRatio` | Classical signal vs. noise. |
| PESQ | `PerceptualEvaluationSpeechQuality` | ITU-T standard, perceptual. |
| STOI | `ShortTimeObjectiveIntelligibility` | Speech intelligibility (0–1). |
| ESTOI | `ExtendedShortTimeObjectiveIntelligibility` | Extension of STOI. |
| SRMR | `SpeechReverberationModulationEnergyRatio` | Reverberation. |
| C50 | `ComplexScaleInvariantSignalDistortionRatio` | Complex-valued SI-SDR. |
| DNSMOS | `DeepNoiseSuppressionMeanOpinionScore` | Learned MOS predictor. |
| NISQA | `NonIntrusiveSpeechQualityAssessment` | Non-intrusive quality. |

PESQ and STOI both wrap external C / Python libraries (`pesq`, `pystoi`) — they're *not* differentiable and *not* fast on GPU. Use them in eval loops, not training loops.

SI-SDR and SDR **are** differentiable. They're commonly used as training losses for source-separation models.

---

## Image metrics

### Pixel-level

| Metric | Class | Use |
|---|---|---|
| PSNR | `PeakSignalNoiseRatio` | Reconstruction quality. |
| SSIM / MS-SSIM | `StructuralSimilarityIndexMeasure`, `MultiScaleStructuralSimilarityIndexMeasure` | Perceptual structural similarity. |
| RMSE@SW | `RootMeanSquaredErrorUsingSlidingWindow` | Localized RMSE. |
| MAE / MSE | (from `regression`) | Pixel-wise error. |
| Spectral Distortion Index | `SpectralDistortionIndex` | Pansharpening evaluation. |
| Spatial Distortion Index | `SpatialDistortionIndex` | Pansharpening evaluation. |
| ERGAS | `ErrorRelativeGlobalDimensionlessSynthesis` | Multispectral fusion. |
| QNR | `QualityNoReferenceMetric` | No-reference quality. |
| UQI | `UniversalImageQualityIndex` | Older quality metric. |

### Feature-level (generative-model evaluation)

| Metric | Class | Notes |
|---|---|---|
| FID | `FrechetInceptionDistance` | Distance between Inception-feature distributions. |
| KID | `KernelInceptionDistance` | MMD-based variant; better small-sample behavior. |
| IS | `InceptionScore` | Older; deprecated for evaluating modern generative models. |
| LPIPS | `LearnedPerceptualImagePatchSimilarity` | Learned perceptual similarity. |
| CLIP Score | `CLIPScore` | Text-image alignment via CLIP. |
| MIFID | `MemorizationInformedFrechetInceptionDistance` | FID adjusted for memorized samples. |
| VIF | `VisualInformationFidelity` | Information-theoretic IQA. |

FID/KID/IS load a feature extractor (Inception V3 by default) on first use. Mind the device. Pass `feature=2048` to use the standard feature dim; smaller dims give worse statistics but faster compute.

### Image practicalities

- **FID is very sensitive to N**. Don't compare FID at 1k samples to FID at 50k samples.
- **Range conventions matter**. Some papers feed `[0, 1]`, some `[-1, 1]`, some `[0, 255]`. PSNR / SSIM in TorchMetrics default to `[0, 1]`; pass `data_range=` if you use something else.
- **LPIPS** has three pretrained backbones (alex, vgg, squeeze). Stick to one when comparing across runs.

---

## Multimodal

`torchmetrics.multimodal` currently includes **CLIP Score** and **CLIP IQA**. CLIP Score is a no-reference text-image alignment metric: it measures cosine similarity between image and prompt embeddings produced by a CLIP model. It's the standard metric for evaluating text-to-image generation today.

---

## Detection

`torchmetrics.detection.MeanAveragePrecision` computes COCO-style mAP for object detection. It accepts the standard list-of-dict format (`boxes`, `scores`, `labels`). Internally it wraps `pycocotools` (or `faster-coco-eval` if installed) and is the canonical implementation.

Trade-offs:

- It is **slow** — the COCO evaluator is not GPU-accelerated.
- It is **memory-heavy** — predictions and targets are kept until `compute()`.
- It is **trusted** — used by the Lightning team to validate detection model releases.

---

## Segmentation

`torchmetrics.segmentation` covers IoU / Jaccard, Dice, Hausdorff distance, generalized dice, and a few clinically-motivated metrics.

The most subtle one is `HausdorffDistance`, which is *worst-case* boundary error — a single rogue pixel can dominate. Use the average symmetric surface distance (ASSD) variant when you want a stable summary.

---

## Video

`torchmetrics.video.VideoMultiMethodAssessmentFusion` (VMAF), and frame-level wrappers around image metrics. VMAF wraps libvmaf; install via `pip install "torchmetrics[video]"`.

---

## Choosing metrics for a generative model

A reasonable evaluation suite for a text-to-image model:

| Concern | Metric |
|---|---|
| Fidelity | FID + KID |
| Text alignment | CLIP Score |
| Sample diversity | KID + recall (Kynkäänniemi et al.) |
| Perceptual quality | LPIPS or LPIPS-based vs reference |
| User study | mean opinion score (out of band) |

For an ASR model:

| Concern | Metric |
|---|---|
| Headline number | WER |
| Character-level | CER |
| Latency-oriented | Custom (RTF) |
| Fairness across accents | Per-bucket WER |

---

## Interview Drill-Down (multi-level follow-ups)

### Q1. BLEU vs ROUGE — when to use which?

> BLEU is **precision-oriented** (n-gram precision with brevity penalty) — used for MT where over-generation is penalized. ROUGE is **recall-oriented** — used for summarization where hitting the reference content matters more than not adding extras.

  **F1.** Why does BLEU have a brevity penalty?

  > Without it, a 1-word output that matches one reference word would have perfect n-gram precision. The brevity penalty `exp(1 - r/c)` punishes too-short outputs.

    **F1.1.** Why does sentence-BLEU give weird numbers compared to corpus-BLEU?

    > BLEU's geometric mean of n-gram precisions is fragile on short text. Smoothing methods (Chen-Cherry, etc.) are needed for sentence-level. Even then, sentence-BLEU is widely considered unreliable. Always prefer corpus-level reporting.

      **F1.1.1.** What metric do you use for *sentence-level* MT evaluation?

      > chrF (`CHRFScore`) is more stable than sentence-BLEU. Or BERTScore for semantic similarity. Modern WMT submissions report COMET (a learned metric) — not in TorchMetrics, but in the same vein.

### Q2. FID at 1k samples vs FID at 50k samples — why differ?

> FID is biased downward at small N — covariance estimates have high variance. The bias decreases as 1/N. Two papers comparing at different N are not comparable.

  **F1.** How do you mitigate small-N bias?

  > Use KID (Kernel Inception Distance). KID's MMD estimator is unbiased even at small N. It's what to report when you genuinely can't run 50k samples.

    **F1.1.** Why isn't KID the default if it's better?

    > FID is the historical literature standard. Papers report FID for backward comparability. KID is a complement, not a replacement.

      **F1.1.1.** What's the right report?

      > Both, with N. "FID@10k = X, KID@10k = Y" is more honest than either alone.

### Q3. Why is BERTScore expensive?

> Each call runs a transformer (default `roberta-large`) over both reference and hypothesis. ~1 GFLOP per token-pair. Bottleneck for any text eval at scale.

  **F1.** How do you make it cheaper without losing fidelity?

  > Distillation: compute BERTScore with `roberta-large` on a held-out set; compute it with a smaller model on the same set; train a calibration regression from small to large. Use the calibrated small-model score in production.

    **F1.1.** Doesn't that lose interpretability?

    > Yes — your numbers no longer compare to literature BERTScore-large. You document this and only use the cheap version for relative comparisons (model-vs-model), not absolute reporting.

      **F1.1.1.** What's the rule for when relative-only is fine?

      > Internal model selection (which checkpoint to ship?) is fine. External benchmarking (paper, blog post, claim of "X is best") needs the canonical metric. The interpretability cost is the cost of the speed-up.

### Q4. PESQ for speech quality — why is it not differentiable?

> PESQ wraps a C reference implementation that includes psychoacoustic filtering, time alignment, and disturbance modeling. None of those steps are differentiable, and the official ITU implementation is not designed to be.

  **F1.** What do you use as a *training* loss for speech enhancement instead?

  > SI-SDR (`ScaleInvariantSignalDistortionRatio`) is differentiable and used widely as the training signal. STOI variants exist as differentiable surrogates (`NegSTOI`).

    **F1.1.** Why train SI-SDR but report PESQ?

    > Different audiences. Researchers train on what optimizes well; product teams report what reviewers know. SI-SDR ↔ PESQ correlation is high but not 1.0 — you're trusting that correlation.

      **F1.1.1.** What if the correlation breaks for your model?

      > Train multi-objective: weighted sum of SI-SDR and a differentiable surrogate for the perceptual signal you actually care about. Or run human MOS evaluation as the final gate.
