---
title: Image quality metrics — interview deep dive
---

# Image quality metrics — interview deep dive

[← Family page](./index.md) · [← Home](../index.md)

---

## Q1. "Why is FID biased for small samples?"

**Answer.** FID fits Gaussians to the real and fake feature distributions then computes Frechet distance. With small N, the empirical covariance matrix is *underestimated* and the trace term `Tr(Σ_r + Σ_g − 2(Σ_r Σ_g)^{1/2})` is too small. Result: FID(N=1000) is systematically lower than the population FID. Standard fix: 50k samples. For < 10k, use **KID** (Kernel Inception Distance) — unbiased estimator.

> **F1.1** "If I have only 5000 generated images, what do I report?"
>
> **Answer.** KID with bootstrap CIs. Or: subsample real images to match (so the bias is symmetric across reports), and report "FID@5k" explicitly. Don't compare FID across different sample counts.

> **F1.2** "Why is the bias *low* and not high?"
>
> **Answer.** The matrix-square-root term is positive-semidefinite; finite-sample covariance underestimation makes both `Σ_r` and `Σ_g` slightly small, but the *cross term* `(Σ_r Σ_g)^{1/2}` shrinks more (it's the geometric mean), so the difference shrinks too. Net: FID estimate is biased low.

---

## Q2. "PSNR vs SSIM vs LPIPS — when to use which?"

**Answer.**
- **PSNR**: dB-scale MSE. Universal, fast, but blind to perception. Use for codec / compression where you have a literal target signal-to-noise.
- **SSIM**: window-wise structural similarity. Better than PSNR for perceived quality but still hand-engineered.
- **LPIPS**: deep-feature distance calibrated against human judgments. Closest to "do humans think these look similar?" Use when *perceptual* quality is the target.

For super-resolution: PSNR + SSIM + LPIPS reported together. PSNR gives you the engineering number; LPIPS the human-experience number; SSIM bridges.

> **F2.1** "Why do papers still report PSNR if LPIPS is better?"
>
> **Answer.** Reproducibility — PSNR is unambiguous (one formula). LPIPS depends on backbone (alex/vgg/squeeze) and version. PSNR remains the cross-paper baseline. The interview answer: report all three; LPIPS for the conclusion, PSNR for the cross-paper sanity check.

---

## Q3. "Generative model on faces — FID went from 12 to 8. Real win?"

**Answer.** Maybe. Three checks:
1. **Sample count consistent**? FID changes with N — re-run at the same N for both models.
2. **Inception preprocessing identical**? Different preprocessing (256×256 vs 299×299, different normalisation) → different FID.
3. **Variance**? Run multiple seeds on each model; report mean ± std. A 4-point drop on top of ±2 std is significant; a 4-point drop on top of ±5 std isn't.

Plus: FID is sensitive to a few hyperparameters (truncation in StyleGAN). State the truncation level alongside the number.

---

## Q4. "Why is Inception Score gameable?"

**Answer.** IS = `exp(E_x [KL(p(y|x) || p(y))])`. High IS rewards (a) sharp class predictions per sample (low entropy of `p(y|x)`) and (b) diverse class predictions across samples (high entropy of `p(y)`). A model that reproduces a few canonical examples per ImageNet class — copying training data — scores extremely high IS without producing novel content. Plus, IS is computed on the *same* network it's optimised against; mode-collapse to "InceptionV3-friendly" images inflates it. **FID has none of these issues** and replaced IS by 2018.

---

## Q5. "Build a metric stack for a diffusion model on faces."

**Answer.**
```python
metrics = {
    "fid":       FrechetInceptionDistance(feature=2048),       # primary
    "kid":       KernelInceptionDistance(subset_size=1000),     # bootstrap-friendly
    "lpips_div": LearnedPerceptualImagePatchSimilarity(...),    # diversity proxy
    "mifid":     MIFID(...),                                    # memorisation check
    "ppl":       PerceptualPathLength(...),                     # latent smoothness
}
```
**Why each**: FID = headline. KID = unbiased small-N. LPIPS pairwise within generated samples = diversity (high LPIPS-pairwise = diverse). MIFID guards against training-data leakage. PPL diagnoses non-smooth latent spaces.

---

## Q6. "PSNR is high but the image looks bad. What's happening?"

**Answer.** PSNR averages MSE pixel-wise — it's blind to *where* errors are. Three failure modes:
- **Blocking artefacts** (codec): pixels are mostly correct, edges at block boundaries are wrong. PSNR averages low. Use PSNR-B.
- **Texture loss** (over-smoothed denoising): MSE goes down, perceived sharpness lost. Use LPIPS.
- **Colour shift on small region** (small support, big visual impact). Use SSIM or LPIPS.

PSNR is necessary but not sufficient.

---

## Q7. "FID's Inception backbone — why is that a problem?"

**Answer.** FID is implicitly *anchored* to whatever an Inception-v3-trained-on-ImageNet considers important. Two issues:
1. **Domain mismatch**: medical imaging, satellite imagery, anime — Inception's features are tuned to natural ImageNet objects. FID on such domains can be misleading.
2. **Adversarial training**: a generator that exploits Inception's blind spots scores low FID without producing visibly better samples.

Workarounds: domain-specific FID using a domain-trained backbone, or perceptual metrics like LPIPS computed against in-domain references.

---

## Q8. "How does TorchMetrics handle DDP for FID?"

**Answer.** State per rank is the running list of feature vectors (real + fake). `_sync_dist` gathers vectors to all ranks (or rank 0). With 50k images at 2048-dim float32, that's 400 MB — large but tractable on modern GPUs. Computing on rank 0 only is the standard pattern.

For training tracking (compute FID every epoch), keep N small (5k) and accept biased numbers; for the paper-quality number, run a final eval with 50k samples.

---

## Q9. "For super-resolution, why is `PSNR + SSIM + LPIPS` the standard triple?"

**Answer.** They cover three different failure modes:
- **PSNR**: pixel fidelity. Catches "wrong pixel values."
- **SSIM**: structural fidelity. Catches "edges and structures preserved."
- **LPIPS**: perceptual fidelity. Catches "looks right to a human."

A model can score well on two and fail the third (e.g., GAN-based SR has high LPIPS performance but low PSNR — produces sharp but pixel-shifted output). Reporting all three is the standard.

---

## Q10. "Hyperspectral imaging — RMSE or SAM?"

**Answer.** **SAM** (Spectral Angle Mapper). Hyperspectral pixels have 100+ channels representing spectral signatures. RMSE measures absolute brightness mismatch; SAM measures *shape* mismatch in spectral space, agnostic to illumination intensity. The same material under different lighting has the same SAM but different RMSE — and material identity is what hyperspectral cares about.

---

## Q11. "Pansharpening evaluation — what's the standard?"

**Answer.** Quad-metric report: ERGAS (relative error across bands, spatial-resolution-aware), SAM (spectral angle), SCC (edge correlation), and either D_λ/D_S/QNR (no-reference) or RMSE (paired). Pansharpening is the unique case where one metric is never enough — spectral and spatial fidelity trade off, so each gets its own number.

---

## Q12. "Memorisation in generative models — how do you detect it?"

**Answer.**
1. **MIFID** — penalises feature-space overlap with training samples.
2. **Nearest-neighbour LPIPS** — for each generated sample, find the closest training image and compute LPIPS. Distribution skewed near zero ⇒ copying.
3. **Pixel-distance histogram** — MSE between generated and nearest training image.

Combined: MIFID for headline, nearest-neighbour LPIPS for sample-level audit.

---

## Q13. "ARNIQA — when do you need a no-reference metric?"

**Answer.** When evaluating *individual* uploads or production samples without paired ground truth: user-uploaded photos, social-media filters, generated content where you don't know what the "right" answer is. Standard reference metrics (PSNR/SSIM/LPIPS) need a target image; ARNIQA/CLIP-IQA give a per-image quality score from a pre-trained scorer.

---

## Q14. "PSNR vs PSNR-B — when do you care?"

**Answer.** Block-coding (JPEG, MPEG, HEVC) produces edge artefacts at the block boundaries. PSNR averages those out and looks fine. PSNR-B adds a blocking-effect penalty so a 30 dB PSNR with bad blocks scores below a 28 dB PSNR with smooth boundaries. Use PSNR-B specifically for codec evaluation.

---

## Q15. "Why is `data_range` such a common bug?"

**Answer.** PSNR formula has `MAX² / MSE`. `MAX` is `data_range`. If your image is in `[0, 1]` and you forget to set `data_range=1.0`, the metric assumes 255 and reports a number 48 dB too high (`20·log10(255/1)`). Always set `data_range` explicitly. Same applies to SSIM. Test on a known-correct example before trusting numbers.

---

## Q16. "Two GANs — model A FID 8.5 ± 0.4, model B FID 7.9 ± 0.6. Which wins?"

**Answer.** Statistically marginal — CI overlap. Three actions:
1. Increase samples per run to reduce variance.
2. More seeds.
3. Co-report KID (lower variance) and LPIPS-pairwise (diversity).

Don't ship "B beats A" on overlapping CIs. The honest report: "FIDs comparable; pending more samples."

---

[← Back to family page](./index.md)
