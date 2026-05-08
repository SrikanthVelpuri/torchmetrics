---
title: Cross-family scenarios — deep dive
---

# Cross-family scenarios — which metric stack for which system?

> Real systems mix metrics from 3-4 families. A recsys is retrieval + classification + regression + fairness. A chatbot is text + classification + retrieval. This page maps **25 production systems → their metric stacks**, with the *why* on every line.

[← Home](../index.md)

For each scenario:
- **Setup** — the system and what it predicts.
- **Metric stack** — what to log.
- **Why each** — what failure mode it catches that others don't.
- **Trap** — the bug a less-experienced team would ship with.

---

## 1. E-commerce search ranker

**Setup.** Query → ranked product list. Graded relevance from human judges + click logs.

**Stack.**
- `RetrievalNormalizedDCG(top_k=10)` — primary KPI.
- `RetrievalNormalizedDCG(top_k=30)` — depth diagnostic.
- `RetrievalMRR(top_k=10)` — "did we put a great match at the top?"
- `RetrievalHitRate(top_k=1)` — interpretable for stakeholders.
- Per-segment NDCG — by query length, language, category.
- Online A/B click-through, conversion, abandonment.

**Why.** NDCG@10 matches the user-facing depth. Depth diagnostic catches "model improves rank 11-30 only." Per-segment guards against tail-segment regression. Online metrics are the truth.

**Trap.** Optimising NDCG offline without an online check. Offline judgments are biased toward what the previous ranker showed; online wins differ.

---

## 2. Recommendation system (carousel)

**Setup.** User → top-K recommended items. Implicit feedback (clicks, dwells).

**Stack.**
- `RetrievalHitRate(top_k=10)` — engagement at all.
- `RetrievalNormalizedDCG(top_k=10)` — rank quality.
- `RetrievalMAP(top_k=10)` — aggregated rank quality with multiple positives.
- IPS-weighted Hit/MRR — debias position.
- Diversity (intra-list LPIPS or category coverage).
- Coverage (fraction of catalog ever shown).
- Per-user-cohort metrics.

**Why.** Implicit feedback is biased; raw Hit@10 over-weights what the previous model showed at top. IPS correction matters. Diversity guards against echo chambers; coverage against catalog stagnation.

**Trap.** Reporting Hit@10 alone — model collapses to the most popular items, scores great, kills long-tail engagement.

---

## 3. Spam filter

**Setup.** Email → {spam, ham}. Cost asymmetry: false-positive (real mail in spam) is 10× worse than false-negative (spam in inbox).

**Stack.**
- `Precision` (with high threshold) — primary.
- `RecallAtFixedPrecision(min_precision=0.99)` — deployment metric.
- `AveragePrecision` — ranking quality.
- `CalibrationError` — for the "report-as-spam" probability.
- Per-sender / per-domain segments.

**Why.** Precision-first because of cost asymmetry. Constrained recall at a precision floor matches the deployment contract.

**Trap.** Reporting accuracy — a 99% accurate spam filter at 99% precision blocks 1% of real mail = thousands of users pissed daily.

---

## 4. Fraud detection

**Setup.** Transaction → {fraud, legit}. Heavy imbalance (0.05% fraud).

**Stack.**
- `AveragePrecision` (AUPRC) — primary, base-rate aware.
- `MatthewsCorrCoef` — single-number model selection.
- `Recall` at the analyst-capacity-defined threshold.
- `Precision@K` where K = analyst daily review capacity.
- Per-segment (merchant, country, channel).
- Bootstrap CIs.

**Why.** AUROC misleads at 0.05% positive rate — AUPRC and MCC are the right single numbers. Precision@K matches "we can only review 50/day."

**Trap.** Reporting AUROC = 0.99 → 0.995 as "great improvement." Both look good; production lift can be zero because both score easy negatives correctly. AUPRC reveals.

---

## 5. Medical imaging cancer screening

**Setup.** Image → {cancer, no-cancer}. Regulator: sensitivity ≥ 95%.

**Stack.**
- `SpecificityAtSensitivity(min_sensitivity=0.95)` — primary, regulatory.
- `AUROC` — overall ranking quality (paper-grade).
- `CalibrationError` — for downstream cost calculations.
- Per-subgroup (sex, age band, scanner manufacturer).
- `BinaryFairness` — disparity across subgroups.
- Bootstrap CIs (test sets are often small).

**Why.** The constrained metric matches regulatory contract. Calibration matters for "expected biopsy load." Per-subgroup catches the skin-tone / scanner failure mode common in medical AI.

**Trap.** Reporting AUROC alone. Regulators reject papers that don't constrain on the actual deployment criterion.

---

## 6. ETA prediction (ride-share)

**Setup.** Trip → ETA in minutes.

**Stack.**
- `MeanAbsoluteError` — primary.
- p99 absolute error — tail metric.
- Bias (mean signed error) — over/under-estimation diagnostic.
- Per-segment (rush hour, airport, weather).
- Per-distance bucket.

**Why.** MAE is interpretable. p99 captures user-frustration outliers. Bias matters: a model that's right on average but always under-estimates ETAs erodes trust.

**Trap.** Reporting RMSE — over-weights one bad outlier; teams chase it instead of the systematic MAE.

---

## 7. Demand forecast (retail)

**Setup.** SKU × store × day → expected units sold.

**Stack.**
- `WeightedMeanAbsolutePercentageError` (wMAPE) — volume-weighted, business KPI.
- `MeanAbsoluteError` per SKU — interpretable.
- Bias (signed) — stockout vs holding-cost diagnostic.
- Naive baseline (last-period) and ratio.
- Per-SKU-cluster (top sellers vs long tail).

**Why.** wMAPE matches P&L impact. Plain MAPE blows up on zero-demand days. Naive baseline ratio is the only honest "are we adding value" check.

**Trap.** Reporting MAPE without wMAPE. Tiny-volume SKUs with noisy demand drive MAPE up; the business doesn't care.

---

## 8. Speech recognition (ASR)

**Setup.** Audio → text.

**Stack.**
- `WordErrorRate` — primary.
- `CharErrorRate` — sub-WER diagnostic for spelling-style errors.
- Per-domain (read speech vs conversational).
- Per-noise-condition (clean vs noisy vs reverb).
- Real-time factor (RTF) — latency budget.

**Why.** WER is industry standard. Per-condition matters more than top-line — a model that's 5% on LibriSpeech-clean and 25% on noisy conditions ships only for the clean use case.

**Trap.** Reporting only LibriSpeech-clean. Production audio has noise, reverb, code-switching.

---

## 9. Machine translation

**Setup.** Source language → target language sentences.

**Stack.**
- `SacreBLEUScore` — cross-paper standard.
- `CHRFScore` — for morphologically-rich languages.
- `BERTScore` — semantic match.
- COMET (external) — learned QE metric.
- Human eval (pairwise) on a sample.

**Why.** BLEU for legacy. chrF for non-English targets. BERTScore for semantic. Human eval for ship/no-ship. No single metric shippable alone.

**Trap.** BLEU 1-point gain shipping without human eval. BLEU optimisations don't always lift human ratings.

---

## 10. Summarisation

**Setup.** Long doc → short summary.

**Stack.**
- `ROUGEScore` (R-1, R-2, R-L) — legacy headline.
- `BERTScore` — semantic similarity.
- QA-based (QuestEval, external) — factuality proxy.
- Length compliance.
- Human eval — overall quality.

**Why.** ROUGE rewards lexical overlap (extractive bias); BERTScore semantic; QA-based catches factual hallucination. Length matters for product UX.

**Trap.** ROUGE-only — abstractive models get punished, extractive models get rewarded — and extractive copy-paste is rarely what users want.

---

## 11. Question answering (extractive)

**Setup.** Question + context → span.

**Stack.**
- `SQuADScore` (F1 + EM) — primary.
- Confidence calibration — `CalibrationError`.
- Abstention quality — F1 only on answered questions.

**Why.** SQuAD F1/EM is the standardised pair. Calibration matters because QA systems often trigger downstream actions on confidence.

---

## 12. Image classification (general)

**Setup.** Image → one of N classes.

**Stack.**
- `Accuracy(top_k=1)` and `Accuracy(top_k=5)`.
- `F1Score(average="macro")` for class-balanced view.
- `ConfusionMatrix` — diagnostic.
- Per-subgroup metrics (e.g., per-camera-model).

**Why.** Top-1 + top-5 + macro F1 + confusion matrix = the standard quartet.

---

## 13. Object detection

**Setup.** Image → list of (box, class, confidence).

**Stack.**
- `MeanAveragePrecision` (COCO style: mAP@[.5:.95]).
- mAP@0.5 — interpretable.
- Per-area-bucket (small, medium, large).
- Per-class AP.
- Latency (p99).

**Why.** COCO mAP is the leaderboard metric. Per-bucket diagnoses size-specific regressions. Latency matters for any deployed system.

---

## 14. Semantic segmentation

**Setup.** Image → per-pixel class.

**Stack.**
- `MeanIoU` — primary.
- Per-class IoU vector.
- Boundary metric (Hausdorff95) for safety-critical.

---

## 15. Medical imaging segmentation (organ)

**Setup.** 3D scan → organ masks.

**Stack.**
- `Dice` per organ — primary.
- `Hausdorff95` per organ (in mm) — boundary.
- Volume difference — bias check.
- NSD at 1mm — surface compliance.

**Why.** Dice is convention; Hausdorff95 surgical relevance; volume difference catches systematic shrinkage.

---

## 16. Generative model (image)

**Setup.** Generates images. Compare against a real distribution.

**Stack.**
- `FrechetInceptionDistance` (FID) at 50k samples — primary.
- `KernelInceptionDistance` (KID) — small-N unbiased complement.
- LPIPS pairwise within generated — diversity proxy.
- `MIFID` — memorisation check.
- Human pairwise eval — gold.

---

## 17. Speech enhancement

**Setup.** Noisy audio → clean audio.

**Stack.**
- `ScaleInvariantSignalNoiseRatio` (SI-SNR) — primary, training-aligned.
- SI-SNR-improvement — what the model adds.
- `PerceptualEvaluationSpeechQuality` (PESQ-WB) — perceptual quality.
- `ShortTimeObjectiveIntelligibility` (ESTOI) — intelligibility.
- `DeepNoiseSuppressionMeanOpinionScore` (DNSMOS) — no-reference for prod.

---

## 18. Source separation

**Setup.** Mixed audio → separated sources.

**Stack.**
- `PermutationInvariantTraining(SI-SDR)` — primary (handles assignment).
- SI-SDR-improvement — gain over mixture input.
- Per-source-class (vocals vs instruments).

---

## 19. Text-to-image generation

**Setup.** Prompt → image.

**Stack.**
- `CLIPScore` — alignment to prompt.
- `FrechetInceptionDistance` — fidelity vs real distribution.
- Compositional benchmarks (T2I-CompBench).
- Human eval.

**Trap.** CLIP score alone — captures coarse alignment, misses spatial / counting / negation.

---

## 20. Video codec / streaming

**Setup.** Source video → compressed bitstream.

**Stack.**
- `VideoMultiMethodAssessmentFusion` (VMAF) — primary.
- PSNR per-frame (mean and percentile) — engineering baseline.
- SSIM — structure check.
- Bitrate budget compliance.

---

## 21. Anomaly detection (time series)

**Setup.** Time-series segments → {anomaly, normal}, often unbalanced.

**Stack.**
- `AveragePrecision` (AUPRC) — primary, imbalance-aware.
- `Precision@K` where K = alert budget.
- Time-to-detect — how soon after onset.
- False-alert rate per day — operational.

---

## 22. Drift detection

**Setup.** Production features vs training features.

**Stack.**
- `JensenShannonDivergence` per feature.
- KS-test p-value (external).
- Per-segment drift.
- Alert threshold tuned to acceptable false-positive rate.

---

## 23. Loan default prediction (compliance-sensitive)

**Setup.** Application → P(default).

**Stack.**
- `AUROC`.
- `CalibrationError` — probabilities feed into pricing.
- `Brier` (NLL on binary outcome).
- `BinaryFairness` ratio across protected attributes.
- Per-segment AUROC (income band, geography).

**Trap.** Optimising AUROC and shipping with bad calibration. Loan pricing miscalibrates → losses.

---

## 24. Self-driving perception

**Setup.** Camera + LIDAR → panoptic segmentation + detection + tracking.

**Stack.**
- `PanopticQuality` (PQ).
- `MeanAveragePrecision` for detection.
- MOTA / IDF1 (external) for tracking.
- Per-distance-bucket (close, medium, far).
- Per-condition (day/night/rain).
- Latency p99.

---

## 25. Code generation (LLM)

**Setup.** Prompt → code.

**Stack.**
- pass@k (functional correctness) — primary.
- Edit distance to ground-truth — for reference-based eval.
- BLEU/CodeBLEU — legacy / cross-paper.
- Compile rate — hygiene.
- Human pairwise on a sample.

**Trap.** Reporting BLEU only — code can be lexically similar but not run. pass@k is the right primary.

---

## Cross-cutting principles

1. **No single metric is enough.** Every system above uses ≥ 3.
2. **Always include a baseline** (naive, BM25, last-value, mean).
3. **Per-segment > top-line** for any system with heterogeneous traffic.
4. **Constrained metrics > unconstrained** when there's a deployment contract (precision floor, sensitivity floor).
5. **Calibration matters** whenever downstream uses *probabilities*.
6. **Online > offline** for the final ship/no-ship; offline is for tracking.
7. **Bootstrap CIs** for small test sets and any cross-model comparison.

---

[← Back to home](../index.md)
