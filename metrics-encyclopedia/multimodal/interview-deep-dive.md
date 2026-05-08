---
title: Multimodal / video / shape — interview deep dive
---

# Multimodal, video, shape — interview deep dive

[← Family page](./index.md) · [← Home](../index.md)

---

## Q1. "Why is CLIP score not enough for text-to-image?"

**Answer.** CLIP score measures *coarse* alignment — "does this image match the caption category." It fails at:
- **Compositional details** ("a red cube on top of a blue sphere") — CLIP can't reliably check spatial relations.
- **Counting** ("five apples") — CLIP can't count.
- **Negation** ("a room without a chair") — CLIP doesn't handle negation well.

So CLIP score + FID + targeted compositional benchmarks (e.g., DrawBench, T2I-CompBench) + human eval together. CLIP alone gives the wrong leaderboard.

---

## Q2. "VMAF vs PSNR for video quality?"

**Answer.** PSNR averages MSE per frame — blind to temporal artefacts (motion judder, frame drops, blocking that varies by content). VMAF fuses multiple perceptual features and is trained to predict MOS. Industry-wide, VMAF correlates with subjective quality at Pearson ~0.9 vs PSNR ~0.5. Netflix uses VMAF to drive their ABR ladder; PSNR drove an older generation of codecs and shows its age.

> **F2.1** "When does VMAF mislead?"
>
> **Answer.** Out of training distribution — animation, screen content, very high resolutions where the original VMAF training set was sparse. Use the right VMAF model variant for the content type (`vmaf_4k_v0.6.1`, `vmaf_v0.6.1`, etc.) and validate against subjective tests for new content categories.

---

## Q3. "How do you evaluate text-to-image generation rigorously?"

**Answer.** Three-axis report:
1. **Fidelity** — FID against a real distribution (or per-prompt with KID for small N).
2. **Alignment** — CLIP score, plus targeted compositional benchmarks.
3. **Diversity** — pairwise LPIPS within generated samples for the same prompt.

Plus human evaluation (pairwise preferences) every release. Single-metric reports get gamed.

---

## Q4. "Procrustes — what's the trick?"

**Answer.** Two point sets that should match up to rigid transformation (e.g., predicted vs ground-truth pose skeletons). Procrustes solves the optimal `(R, t, s)` (rotation, translation, scale) that aligns one to the other in least-squares sense; the residual is the Procrustes distance. Used because most pose datasets have arbitrary global frames — alignment is a nuisance you should remove.

---

## Q5. "How does CLIP score's temperature affect interpretation?"

**Answer.** CLIP's contrastive loss uses a temperature scaling. The metric is `100 · cos_sim`. So a score of 30 means cosine similarity 0.3 — moderate alignment. Don't read 30 as "30%" or "30 dB"; it's just `cos · 100`. Compare deltas, not absolutes; baselines vary widely (~25-35 for moderate prompts on standard models).

---

## Q6. "How does TorchMetrics handle DDP for CLIP score?"

**Answer.** State per rank: list of CLIP scores. CLIP forward pass runs per rank (each rank embeds its own batch). `_sync_dist` aggregates the scalar list — cheap. The compute is dominated by the CLIP forward, not the metric.

---

## Q7. "VMAF is too slow for online use. What's the workaround?"

**Answer.**
1. **Sub-sampling**: VMAF on every Nth frame is unbiased for the mean.
2. **Faster models**: `vmaf_v0.6.1neg` is faster than full VMAF.
3. **Offline ABR optimisation**: pre-compute VMAF for the bitrate ladder offline; serve PSNR-driven ABR online with offline VMAF as the design KPI.

---

[← Back to family page](./index.md)
