---
title: Audio metrics — interview deep dive
---

# Audio metrics — interview deep dive

[← Family page](./index.md) · [← Home](../index.md)

---

## Q1. "SNR vs SI-SNR — when does it matter?"

**Answer.** Pure SNR penalises predictions at the wrong volume — even if perceptually identical. SI-SNR removes the optimal scaling first, so it measures *shape* fidelity, not amplitude. Two consequences:
- **Training**: SI-SNR loss prevents the model from being penalised for producing scale-correct-but-shape-perfect outputs.
- **Evaluation**: SI-SNR is the standard for speech enhancement reporting (WHAMR!, DNS Challenge) precisely because amplitude calibration is a separate downstream problem.

> **F1.1** "Why does optimal-scaling matter for time-domain enhancement?"
>
> **Answer.** Many enhancement nets output normalised waveforms (logits / tanh). The absolute amplitude is determined by an output rescaling that's not the model's job. Penalising amplitude in the loss double-counts and slows convergence.

---

## Q2. "What is PIT and why is it needed?"

**Answer.** Permutation-Invariant Training. In source separation, the network outputs N waveforms but doesn't know which output should match which target — assignment is arbitrary. PIT tries all `N!` permutations of assignment, picks the one that minimises the loss (or maximises the SI-SNR), and trains on that. Without PIT, the model can't learn — gradient flips as the optimal assignment changes per batch.

> **F2.1** "What's the cost of PIT?"
>
> **Answer.** `O(N!)` per batch where N = number of sources. For N=2: 2 forward orderings. For N=3: 6. For N=8: 40320 — too expensive. For larger N, use **uPIT** (utterance-level PIT) which assigns once per utterance, or **PIT with Hungarian assignment** (`O(N³)`).

---

## Q3. "PESQ is at 3.2 — is the model good?"

**Answer.** Depends on baseline and condition. PESQ-WB scale: noisy speech ~1.5-2.0; well-enhanced ~3.0-3.5; clean reference itself ~4.5. PESQ 3.2 with input PESQ 1.8 is a strong improvement; PESQ 3.2 starting from input 3.1 is marginal. Always report **PESQ-improvement** alongside the absolute value.

> **F3.1** "Why is PESQ deprecated?"
>
> **Answer.** ITU replaced PESQ with POLQA (P.863) in 2011 — POLQA handles wideband and super-wideband better, plus perceptual model improvements. Most papers still report PESQ for legacy comparison. POLQA is licensed; PESQ is in TorchMetrics for compatibility.

---

## Q4. "STOI vs PESQ — what do each measure?"

**Answer.**
- **STOI**: speech *intelligibility* — % words a listener will understand. Predicts task success.
- **PESQ**: speech *quality* — listener's overall liking. Predicts user satisfaction.

A heavily processed signal can have high STOI (still understandable) but low PESQ (sounds robotic). Always co-report both for speech enhancement.

---

## Q5. "Build a metric stack for a speech-enhancement model."

**Answer.**
```python
metrics = MetricCollection({
    "si_snr":     ScaleInvariantSignalNoiseRatio(),
    "si_snr_imp": SI_SNR_Improvement_wrapper,             # output - input
    "pesq":       PerceptualEvaluationSpeechQuality(fs=16000, mode="wb"),
    "stoi":       ShortTimeObjectiveIntelligibility(fs=16000, extended=True),
    "dnsmos":     DeepNoiseSuppressionMeanOpinionScore(fs=16000),
})
```
**Why each**: SI-SNR for training and headline. SI-SNR-improvement for the gain over noisy input. PESQ for telecom-standard quality. ESTOI for intelligibility. DNSMOS for no-reference (validates that the model didn't optimise for the reference at the cost of perceptual quality).

---

## Q6. "Source separation — what's the standard reporting?"

**Answer.** SI-SDR with PIT, plus SI-SDR-improvement (`SI-SDR(output) − SI-SDR(input mixture)`). Improvement is what the model contributes; the absolute SI-SDR depends heavily on the input mixture difficulty. Always co-report so reviewers see both.

> **F6.1** "Why SI-SDR and not SDR?"
>
> **Answer.** SI-SDR has a clean optimal-scaling decomposition that doesn't double-count amplitude errors — same reason SI-SNR > SNR for speech enhancement.

---

## Q7. "DNSMOS vs PESQ — when to pick which?"

**Answer.** PESQ requires a clean reference. DNSMOS doesn't. In production monitoring you usually don't have references. Use DNSMOS for live-monitoring; PESQ for offline benchmarking with held-out clean signals.

The practical move at scale: train and validate with PESQ (you have references), monitor production with DNSMOS (you don't).

---

## Q8. "What sample rate do these metrics run at?"

**Answer.** Critical:
- **PESQ-NB**: 8 kHz.
- **PESQ-WB**: 16 kHz.
- **STOI**: 10 kHz internally (resampled if you provide other rates).
- **SI-SNR / SI-SDR**: any (no internal resampling).
- **DNSMOS**: 16 kHz.
- **NISQA**: 16 kHz.

Mismatched sample rate causes wrong numbers — the metric runs but means something different. Always pin and document.

---

## Q9. "Source separation quality regressed by 1 dB SI-SDR. Diagnosis?"

**Answer.** Drill-down ladder:
1. **Per-mixture-difficulty buckets** (low SNR vs high) — has it regressed everywhere or only on hard mixtures?
2. **Per-source-class** (male vs female speaker, music vs speech) — class-specific regression?
3. **PIT assignment stability** — has the optimal assignment shifted (model now matches differently)?
4. **Listening test on N=20** — is the regression *audible*?

A 1 dB SI-SDR shift below the audible threshold isn't worth blocking; above it (often 1-2 dB) is.

---

## Q10. "Why are no-reference audio metrics like NISQA needed?"

**Answer.** Production deployment doesn't have clean references — you're processing real-time user audio. PESQ requires the clean signal, so it's offline-only. NISQA / DNSMOS are trained networks that predict MOS from a single noisy/enhanced signal. Used for: A/B testing in the wild, daily quality monitoring, regression detection in production. Trade-off: predicted MOS has wider CI than PESQ, so the offline benchmark stays.

---

## Q11. "How does TorchMetrics handle DDP for audio metrics?"

**Answer.** Most are stateless within a batch and aggregate via mean — `(sum, count)` tuples synced via `all_reduce(SUM)`. Cheap. Heavy ones (PESQ, NISQA) call into Python (PESQ) or a TF/PyTorch backbone (NISQA) — those run on each rank in parallel; TorchMetrics still aggregates the resulting scalars cheaply.

---

## Q12. "What's the most common audio-metric bug?"

**Answer.** Wrong sample rate. Resampling with the wrong filter (low-quality default) injects aliasing that all the perceptual metrics catch — your "model regressed" turns out to be the resampler. Always use a high-quality resampler (`sox` or `librosa.resample(res_type='kaiser_best')`) for evaluation.

---

[← Back to family page](./index.md)
