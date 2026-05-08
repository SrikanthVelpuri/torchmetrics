---
title: Image quality metrics — deep dive
---

# Image quality metrics — deep dive

> Image quality metrics split into three groups by what they compare:
>
> - **Reference-based** (have a ground-truth image): SSIM, MS-SSIM, PSNR, LPIPS, DISTS, VIF, UQI, SAM, ERGAS, RASE, RMSE-SW, SCC, PSNRB, TV.
> - **Distribution-based** (compare *sets* of images): FID, KID, MIFID, IS, perceptual path length.
> - **No-reference** (single image, no ground truth): ARNIQA, CLIP-IQA.
>
> Picking wrong gives meaningless numbers — distribution metrics on paired images, reference metrics on generative models that have no paired truth.

[← Home](../index.md) · [Interview deep-dive →](./interview-deep-dive.md)

## Reference-based: the canonical pair (truth + restored/predicted)

### PeakSignalNoiseRatio / PSNR (`tm.image.PeakSignalNoiseRatio`)

**What it computes.** `10 · log10(MAX² / MSE)`. dB-scale signal-to-noise.

**Intuition.** MSE on a log scale, expressed as decibels. Higher = closer to truth. PSNR = 30 dB ≈ "obviously distorted." 40 dB ≈ "indistinguishable to most viewers." 50+ dB ≈ "near-perfect."

**Range / direction.** `[0, ∞)` (∞ when MSE = 0). Higher better.

**When to use.** Image compression, denoising. **Not** correlated well with human perception.

**Real-world scenario.** JPEG compression quality reporting. PSNR is the universal standard but blind to perceptual quality — the eye sees blocking artefacts that PSNR averages out.

**Pitfalls.**
- `data_range` argument: must match the input scale (1.0 if [0,1], 255 if [0,255]). Default depends on dtype — pin it explicitly.

---

### StructuralSimilarityIndexMeasure / SSIM (`tm.image.StructuralSimilarityIndexMeasure`)

**What it computes.** Window-wise comparison of luminance, contrast, and structure.

**Intuition.** Closer to perception than PSNR — captures structural similarity, not just pixel-by-pixel. Rewards preserving edges.

**Range / direction.** `[-1, 1]` (`[0, 1]` for natural images). Higher better.

**Real-world scenario.** Super-resolution / denoising leaderboards: SSIM is the second column after PSNR. Better correlates with subjective quality.

---

### MS-SSIM (`tm.image.MultiScaleStructuralSimilarityIndexMeasure`)

**What it computes.** SSIM over multiple resolution scales, weighted product.

**Intuition.** Captures structural similarity across scales — small details *and* large structures.

**Real-world scenario.** Image-quality datasets (LIVE, TID2013) — MS-SSIM is the SSIM successor with better human-quality correlation.

---

### LPIPS — Learned Perceptual Image Patch Similarity (`tm.image.LearnedPerceptualImagePatchSimilarity`)

**What it computes.** Distance between deep features of two images, where the feature distance is calibrated against human perceptual judgments.

**Intuition.** Trained network whose distance correlates with "do humans see these as similar?" The most perceptually-aligned reference metric.

**Range / direction.** `[0, ∞)`. **Lower better.**

**Real-world scenario.** GAN training: PSNR/SSIM stay flat while LPIPS measures real perceptual progress. Standard reporting metric for super-resolution.

**Pitfalls.**
- Loads a backbone (`alex`, `vgg`, `squeeze`); pin the choice across reports.
- Small inputs (< 64×64) may fail the patch sampling.

---

### DISTS — Deep Image Structure and Texture Similarity (`tm.image.DeepImageStructureAndTextureSimilarity`)

**What it computes.** Combines deep-feature texture statistics with structure similarity. Newer than LPIPS, often better correlated with humans.

---

### VisualInformationFidelity / VIF (`tm.image.VisualInformationFidelity`)

**What it computes.** Information-theoretic measure: mutual information between reference and distorted image as a fraction of reference's information.

**Range / direction.** `[0, 1]`. Higher better.

**Real-world scenario.** Image-quality assessment competitions; together with VMAF used for video.

---

### UniversalQualityIndex / UQI (`tm.image.UniversalImageQualityIndex`)

**What it computes.** Decomposes correlation × luminance × contrast similarity into one bounded score.

**Real-world scenario.** Predecessor of SSIM; rarely the primary metric today but still in legacy IQA pipelines.

---

### SpectralAngleMapper / SAM (`tm.image.SpectralAngleMapper`)

**What it computes.** Per-pixel angle between predicted and reference *spectral vectors* (across channels).

**Intuition.** For multispectral / hyperspectral imagery — measures *colour fidelity* directly, agnostic to brightness scaling.

**Real-world scenario.** Hyperspectral imaging (remote sensing, mineralogy): SAM is the canonical metric — independent of illumination intensity.

---

### ErrorRelativeGlobalDimensionlessSynthesis / ERGAS (`tm.image.ErrorRelativeGlobalDimensionlessSynthesis`)

**What it computes.** Multispectral remote-sensing fidelity: relative RMSE across bands, scaled by spatial resolution ratio.

**Real-world scenario.** Pansharpening evaluation in satellite imagery — combine high-res panchromatic with low-res multispectral and ERGAS measures fidelity to ideal.

---

### RelativeAverageSpectralError / RASE

**What it computes.** Average relative error over spectral bands. Companion to ERGAS for multispectral.

---

### SpatialCorrelationCoefficient / SCC (`tm.image.SpatialCorrelationCoefficient`)

**What it computes.** Edge-detected version of correlation — agreement on where the *edges* are.

**Real-world scenario.** Pansharpening — SCC measures whether edges in the synthesised image align with the reference panchromatic.

---

### RMSE-SW (`tm.image.RootMeanSquaredErrorUsingSlidingWindow`)

**What it computes.** RMSE in a sliding window, averaged. Captures local error rather than global.

---

### TotalVariation / TV (`tm.image.TotalVariation`)

**What it computes.** Sum of `|∇I|` — total amount of "edge content" in the image.

**Real-world scenario.** Used as a regulariser in inverse-problems (denoising, deblurring). Not really a *quality* metric — more a smoothness diagnostic.

---

### PSNR-B (`tm.image.PeakSignalNoiseRatioWithBlockedEffect`)

**What it computes.** PSNR + a blocking-artefact penalty.

**Real-world scenario.** Block-based codec evaluation (JPEG, HEVC) — PSNR misses block boundaries; PSNR-B catches them.

---

### Spectral distortion metrics for pansharpening

- **D_λ (`tm.image.SpectralDistortionIndex`)** — spectral distortion in pansharpened multispectral.
- **D_S (`tm.image.SpatialDistortionIndex`)** — spatial distortion in pansharpened.
- **QNR (`tm.image.QualityWithNoReference`)** — combination of D_λ and D_S, no-reference quality.

These are remote-sensing-specific and rarely used outside that domain.

---

## Distribution-based: the GAN/diffusion family

These take *sets* of images (real and generated) and compare distributions in feature space. **They do not need paired ground truth.**

### FrechetInceptionDistance / FID (`tm.image.FrechetInceptionDistance`)

**What it computes.** Distance between Gaussian-fitted feature distributions of real and generated sets, in Inception-v3 feature space: `FID = ||μ_r − μ_g||² + Tr(Σ_r + Σ_g − 2(Σ_r Σ_g)^{1/2})`.

**Intuition.** Single number measuring "do generated images look like real ones?" The canonical generative-model KPI.

**Range / direction.** `[0, ∞)`. **Lower better.** State-of-the-art on FFHQ ~2.0; ImageNet ~3.0.

**When to use.** Any generative model with a real-image distribution to compare against.

**Real-world scenario.** GAN/diffusion paper leaderboard: FID is *the* number. 50k samples is the standard sample count.

**Pitfalls.**
- **FID is biased low for small N.** With N=1000, FID looks better than the truth. Always use ≥ 10k samples; 50k is standard.
- Inception network normalisation: TorchMetrics does the standard ImageNet normalisation; if you preprocessed differently, normalise before passing in.
- FID has high variance. Always report mean ± std over multiple seeds.

**Code.**
```python
from torchmetrics.image.fid import FrechetInceptionDistance
fid = FrechetInceptionDistance(feature=2048, normalize=True)
fid.update(real_images, real=True)
fid.update(fake_images, real=False)
fid.compute()
```

---

### KernelInceptionDistance / KID (`tm.image.KernelInceptionDistance`)

**What it computes.** Maximum Mean Discrepancy with a polynomial kernel on Inception features.

**Intuition.** Like FID but unbiased and works on smaller sample sizes (~1000).

**Range / direction.** `[0, ∞)`. Lower better.

**Real-world scenario.** When generating few samples (e.g., conditional models with few examples per class), KID is the right choice; FID is too biased.

---

### MIFID — Memorization-Informed FID (`tm.image.MemorizationInformedFrechetInceptionDistance`)

**What it computes.** FID with a memorisation penalty — punishes models that simply copy training images.

**Real-world scenario.** Generative-model competitions on Kaggle (e.g., Doodle-to-Cat) where copying training data is technically valid but uninteresting. MIFID rewards real generation.

---

### InceptionScore / IS (`tm.image.InceptionScore`)

**What it computes.** `exp(E_x [KL(p(y|x) || p(y))])` — measures classifier confidence and diversity simultaneously.

**Range / direction.** `[1, # classes]`. Higher better.

**Real-world scenario.** Older GAN papers; largely superseded by FID. Mostly informational. **IS is biased and gameable** — a model that produces sharp samples for known classes scores high regardless of fidelity.

---

### PerceptualPathLength / PPL (`tm.image.PerceptualPathLength`)

**What it computes.** Average perceptual distance along latent-space interpolation paths.

**Intuition.** "Is the model's latent space smooth?" Used in StyleGAN family.

---

## No-reference

### ARNIQA — Audio Resilience and Natural Image Quality Assessment (`tm.image.ARNIQA`)

**What it computes.** Single-image quality score from a self-supervised pre-trained scorer.

**Range / direction.** Score (often `[0, 1]`). Higher better.

**Real-world scenario.** Quality monitoring of user-uploaded content where there's no clean reference — pick out blurry/over-compressed images at upload.

---

## Quick-reference: which image metric for which scenario?

| Scenario | Primary | Secondary |
|---|---|---|
| Compression / codec | PSNR | SSIM, MS-SSIM |
| Denoising / SR (paired) | PSNR + LPIPS | DISTS |
| GAN / diffusion (unpaired) | FID | KID |
| Few-sample generation | KID | LPIPS-distance |
| Memorisation check | MIFID | nearest-neighbour LPIPS |
| Hyperspectral / remote sensing | SAM | ERGAS, RASE |
| Pansharpening | ERGAS | D_λ, D_S, QNR, SCC |
| Block-coding artefacts | PSNR-B | SSIM |
| User-uploaded quality screening | ARNIQA / CLIP-IQA | — |
| Latent smoothness | Perceptual Path Length | — |

---

[Continue → interview deep-dive](./interview-deep-dive.md)
