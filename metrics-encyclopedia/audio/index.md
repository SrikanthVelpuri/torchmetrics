---
title: Audio metrics — deep dive
---

# Audio metrics — deep dive

> Audio metrics split into three groups:
> - **Reference-based perceptual** (PESQ, STOI) — psychoacoustic scores against clean reference.
> - **SNR family** (SNR, SDR, SI-SNR, SI-SDR, C-SI-SNR) — signal-to-noise ratio variants.
> - **No-reference quality** (DNSMOS, NISQA, SRMR) — single-signal quality estimators.
> - **Source separation accounting**: PIT (Permutation-Invariant Training).

[← Home](../index.md) · [Interview deep-dive →](./interview-deep-dive.md)

## Reference-based perceptual

### PESQ — Perceptual Evaluation of Speech Quality (`tm.audio.PerceptualEvaluationSpeechQuality`)

**What it computes.** ITU-T P.862 standard: psychoacoustic comparison between reference and degraded speech. Outputs a MOS-LQO (Mean Opinion Score, Listening Quality Objective) value.

**Range / direction.** `[1, 4.5]` (narrowband) or `[1, 4.5]` (wideband). Higher better.

**When to use.** Speech enhancement, codec evaluation. The historical telecom standard.

**Real-world scenario.** VoIP codec quality assessment. PESQ is the regulatory and industrial standard for "is this codec acceptable?"

**Pitfalls.**
- PESQ requires 8 kHz (NB) or 16 kHz (WB) input — resample first.
- Has been *deprecated* by ITU in favour of POLQA, but TorchMetrics ships PESQ for legacy comparison.

---

### STOI — Short-Time Objective Intelligibility (`tm.audio.ShortTimeObjectiveIntelligibility`)

**What it computes.** Correlation of envelope time-frequency features between reference and degraded speech.

**Range / direction.** `[0, 1]`. Higher better. Maps to speech intelligibility (% of words understood).

**Real-world scenario.** Hearing-aid algorithm evaluation; speech-enhancement targeting *intelligibility* (vs PESQ which targets quality).

**Pitfalls.**
- Computed at 10 kHz internally — resample to 10 kHz.
- ESTOI (extended) handles modulated noise better; some libs only ship the original STOI. TorchMetrics has `extended=True` flag.

---

## SNR family

### SignalNoiseRatio / SNR (`tm.audio.SignalNoiseRatio`)

**What it computes.** `10 · log10(||signal||² / ||noise||²)` where noise = prediction − target.

**Range / direction.** `(-∞, ∞)`. Higher better.

**Real-world scenario.** Generic audio quality where you have a clean reference. Low-cost, high-interpretability.

**Pitfalls.**
- Sensitive to scaling: a 2× quieter prediction has a different SNR even if perceptually identical. Use SI-SNR/SI-SDR for scale-invariance.

---

### ScaleInvariantSignalNoiseRatio / SI-SNR (`tm.audio.ScaleInvariantSignalNoiseRatio`)

**What it computes.** SNR after optimal scalar rescaling of the prediction to match the target.

**Intuition.** Removes the trivial "predict at the wrong volume" failure mode that pure SNR penalises. Standard for speech enhancement.

**Range / direction.** `(-∞, ∞)`. Higher better.

**Real-world scenario.** Speech-enhancement training loss: train with negative SI-SNR; report SI-SNR-improvement (output minus input SI-SNR) as the headline.

---

### SignalDistortionRatio / SDR (`tm.audio.SignalDistortionRatio`)

**What it computes.** Decomposes the prediction into target + interference + artifacts; SDR is the ratio of target energy to (interference + artifacts).

**Real-world scenario.** Source separation evaluation (e.g., separating two speakers).

---

### ScaleInvariantSignalDistortionRatio / SI-SDR (`tm.audio.ScaleInvariantSignalDistortionRatio`)

**What it computes.** SDR with optimal rescaling, similar to SI-SNR but using SDR's decomposition.

**Real-world scenario.** Source separation benchmarks (WHAM!, LibriMix). SI-SDR is the canonical metric.

---

### ComplexScaleInvariantSignalNoiseRatio / C-SI-SNR

**What it computes.** SI-SNR computed in the complex spectrogram domain.

**Real-world scenario.** Complex-valued speech enhancement (where phase matters). Often used alongside time-domain SI-SNR.

---

### PermutationInvariantTraining / PIT (`tm.audio.PermutationInvariantTraining`)

**What it computes.** For multi-source separation: tries every permutation of predicted-to-target source assignment, picks the one minimising the wrapped metric (e.g., SI-SNR).

**Intuition.** When separating 2+ sources, the model doesn't know which output corresponds to which target. PIT picks the optimal assignment per sample so the metric isn't punished for arbitrary ordering.

**Real-world scenario.** Two-speaker source separation. Forward through the model gives output1, output2. PIT(SI-SNR) finds whether (output1→target1, output2→target2) or the swap gives better SI-SNR, uses that.

**Code.**
```python
from torchmetrics.audio import PermutationInvariantTraining
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
pit = PermutationInvariantTraining(scale_invariant_signal_noise_ratio,
                                   eval_func="max")
pit(preds, target)  # both shape (B, sources, time)
```

---

## No-reference quality

### DNSMOS — Deep Noise Suppression Mean Opinion Score (`tm.audio.DeepNoiseSuppressionMeanOpinionScore`)

**What it computes.** A trained network that predicts MOS from a single noisy/enhanced signal — no reference needed.

**Real-world scenario.** DNS Challenge (Microsoft) — DNSMOS is the official metric. Used for production speech-enhancement quality monitoring where clean reference doesn't exist.

---

### NISQA — Non-Intrusive Speech Quality Assessment (`tm.audio.NonIntrusiveSpeechQualityAssessment`)

**What it computes.** Deep model predicts MOS, plus per-aspect sub-scores (noisiness, coloration, discontinuity, loudness).

**Real-world scenario.** Production telephony quality monitoring at scale — no reference signal available; NISQA gives per-aspect diagnostics.

---

### SRMR — Speech-to-Reverberation Modulation energy Ratio (`tm.audio.SpeechReverberationModulationEnergyRatio`)

**What it computes.** Ratio of modulation energy in speech-relevant vs reverb-dominated bands.

**Real-world scenario.** Dereverberation algorithm evaluation — measures reverb suppression specifically, not generic noise.

---

## Quick-reference: which audio metric for which scenario?

| Scenario | Primary | Secondary |
|---|---|---|
| Speech enhancement (paired) | SI-SNR-improvement | PESQ, STOI |
| Source separation | SI-SDR (with PIT) | SDR |
| Speech codec quality | PESQ-WB | POLQA (external) |
| Hearing-aid intelligibility | ESTOI | STOI |
| No-reference monitoring | DNSMOS, NISQA | — |
| Dereverberation | SRMR | SI-SDR |
| Voice cloning / TTS quality | DNSMOS | MCD (external) |

---

[Continue → interview deep-dive](./interview-deep-dive.md)
