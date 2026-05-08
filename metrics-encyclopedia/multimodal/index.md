---
title: Multimodal / video / shape metrics — deep dive
---

# Multimodal, video, shape — deep dive

> Three small families bundled:
> - **Multimodal**: CLIP-based metrics that score (image, text) joint quality.
> - **Video**: VMAF for video quality.
> - **Shape**: Procrustes for point-set similarity.

[← Home](../index.md) · [Interview deep-dive →](./interview-deep-dive.md)

---

## Multimodal

### CLIPScore (`tm.multimodal.CLIPScore`)

**What it computes.** Cosine similarity between CLIP image and text embeddings, multiplied by a temperature constant.

**Intuition.** "Does this caption match this image?" Pre-trained CLIP captures broad image-text alignment.

**Range / direction.** Roughly `[0, 100]` after CLIP's temperature scaling. Higher better.

**Real-world scenario.** Text-to-image generation evaluation: does the generated image match the prompt? CLIP score is the standard automatic metric (alongside FID for fidelity).

**Pitfalls.**
- **CLIP has known biases** (gender, race) and **doesn't reward attention to detail**. A model that produces vaguely correct images scores high. Co-report human evaluation.
- Backbone matters (`openai/clip-vit-large-patch14`, etc.).

---

### CLIPImageQualityAssessment / CLIP-IQA (`tm.multimodal.CLIPImageQualityAssessment`)

**What it computes.** CLIP-based no-reference image quality: cosine similarity to "good photo" prompts vs "bad photo" prompts.

**Range / direction.** `[0, 1]`. Higher better.

**Real-world scenario.** No-reference quality monitoring of generated or user-uploaded content where ARNIQA isn't available.

---

### LVE — Label-and-Visual Embedding (`tm.multimodal.LVE`)

**What it computes.** Multimodal alignment metric using both label and visual embeddings.

---

## Video

### VMAF — Video Multimethod Assessment Fusion (`tm.video.VideoMultiMethodAssessmentFusion`)

**What it computes.** Netflix's perceptual video quality metric: fusion of multiple per-frame features (VIF, ADM, motion) via a trained model that outputs a MOS-aligned score.

**Range / direction.** `[0, 100]`. Higher better.

**Real-world scenario.** Video codec evaluation, ABR ladder optimisation, perceptual quality monitoring at scale. The streaming-industry standard. PSNR per-frame averaged is a much weaker proxy.

**Pitfalls.**
- Different VMAF models for different displays (4k TV vs phone) — pick the right model for the deployment context.
- Computationally expensive — fine for offline benchmarking, expensive for real-time.

---

## Shape

### ProcrustesAnalysis (`tm.shape.ProcrustesAnalysis`)

**What it computes.** Distance between two point sets after optimal translation, rotation, and scaling alignment.

**Real-world scenario.** Pose estimation, mesh comparison: two skeletons of the same person should agree up to translation/rotation/scale. Procrustes distance measures the residual after that alignment.

---

## Quick-reference

| Scenario | Primary | Secondary |
|---|---|---|
| Text-to-image (generative) | CLIP Score | FID, human eval |
| Image-text retrieval | CLIP Score | Recall@K |
| No-reference image quality | CLIP-IQA | ARNIQA |
| Video codec / streaming | VMAF | PSNR, SSIM |
| Pose / mesh comparison | Procrustes distance | per-joint MPJPE |

---

[Continue → interview deep-dive](./interview-deep-dive.md)
