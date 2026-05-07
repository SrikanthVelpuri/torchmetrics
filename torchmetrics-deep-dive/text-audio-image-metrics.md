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
