---
title: Metric Decision Trees
nav_order: 23
---

# Visual Decision Trees — Which Metric for X?

Print these. Stick them on your wall. When an interviewer asks "which metric would you use?" — you don't think, you walk down the tree.

---

## Tree 1 — The master flowchart

```text
                              START
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        IS THE OUTPUT?    IS THE OUTPUT?   IS THE OUTPUT?
         a CLASS LABEL    a CONTINUOUS     a RANKED LIST?
              │             VALUE?              │
              ▼                ▼                ▼
       see Tree 2          see Tree 3       see Tree 4
       (Classification)    (Regression)     (Retrieval)


              ▼                ▼                ▼
       IS THE OUTPUT?    IS THE OUTPUT?   IS THE OUTPUT?
        TEXT?            AN IMAGE?         AUDIO?
              │             │                  │
              ▼             ▼                  ▼
       see Tree 5      see Tree 6         see Tree 7
       (Text)          (Image)            (Audio)
```

---

## Tree 2 — Classification

```text
                    CLASSIFICATION OUTPUT
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
     BINARY            MULTICLASS           MULTILABEL
        │                   │                   │
        ▼                   ▼                   ▼
   Imbalanced?         Balanced?           Many labels?
   ┌────┴────┐         ┌────┴────┐         ┌────┴────┐
   YES       NO        YES       NO        YES       NO
    │         │         │         │         │         │
    ▼         ▼         ▼         ▼         ▼         ▼
 Use AP    Use F1   Use Top-1  Use F1   Use macro-F1  Use F1
 (★safe)  + AUROC  + Top-5    macro    over POS-       (Hamming
                              + per-    SUPPORT        if exact
                              class     subset only    match too
                                                       brutal)
                                        + worst-K
                                        per-class

   Need calibration?  →  + CalibrationError
   Need fairness?     →  + BinaryFairness / per-group F1
   Need an operating point with hard precision floor?
                      →  RecallAtFixedPrecision
   Need an operating point with hard recall floor?
                      →  PrecisionAtFixedRecall
```

---

## Tree 3 — Regression

```text
                    REGRESSION OUTPUT
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
      Is target's range         Is target probabilistic?
      bounded / interpretable?   (you predict a distribution)
              │                           │
        ┌─────┴─────┐                     ▼
        YES          NO                Use CRPS or
        │            │                 quantile loss
        ▼            ▼
   Care about      Heavy-tailed
   outliers?       errors?
   ┌────┴────┐     ┌────┴────┐
   YES        NO   YES         NO
    │          │    │            │
    ▼          ▼    ▼            ▼
   MAE        MSE  Huber/       MSE
            (RMSE   Log-cosh    + RMSE
             too)   (custom)

   Need cross-series comparability?  →  wMAPE (NOT MAPE if y can be 0)
   Need percent in [0, 200%]?         →  SMAPE (with floor on near-zero)
   Need correlation, not error?       →  Tree 3a
   Need calibration on probability?   →  Brier (= MSE on probs) + ECE
```

### Tree 3a — Correlations

```text
                    CORRELATION TYPE
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
     Linear?           Monotonic?          Agreement?
        │                   │                   │
        ▼                   ▼                   ▼
   Pearson           Spearman              Lin's CCC
                     (ranks; list state)   (Concordance)
        │                   │
        ▼                   ▼
   Streaming OK       Use compute_on_cpu
   (Welford's algo)   for huge eval

   Rank concordance?  →  Kendall τ (slow; O(n²))
```

---

## Tree 4 — Retrieval / Ranking

```text
                    RETRIEVAL TASK
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
      First-stage retrieval?          Reranker?
      (catalog → candidates)          (candidates → top-K)
            │                               │
            ▼                               ▼
      Recall @ K_large            ┌────────┼────────┐
      (K = 100/500/1000)          ▼        ▼        ▼
                              Binary    Graded   Mixed
                              relevance relevance
                                  │       │
                                  ▼       ▼
                                 MAP     NDCG
                                 MRR     NDCG@K

   Does the user only see top-K?   →  Truncate metric @ K
   Are queries weighted by traffic? →  Weighted aggregation (custom)
   Are queries cold-start?          →  empty_target_action='skip'
   Need calibration of P(relevant)? →  add CalibrationError
```

---

## Tree 5 — Text

```text
                    TEXT TASK
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
    Translation        Summarization         ASR
        │                   │                   │
        ▼                   ▼                   ▼
   Corpus eval?         Lexical or          Word-level
        │               Semantic?           or char-level?
   ┌────┴────┐          ┌────┴────┐         ┌────┴────┐
   YES        NO        Lexical   Semantic  Word      Char
    │          │           │         │       │         │
    ▼          ▼           ▼         ▼       ▼         ▼
   BLEU       chrF       ROUGE     BERTScore WER       CER
   (corpus    (sentence- (1, 2, L)
    only)     stable)

   LM perplexity?    →  Perplexity (token log-probs)
   QA factuality?    →  exact-match + F1 over tokens
   RAG retrieval?    →  RetrievalRecall@K + RetrievalNDCG (Tree 4)
```

---

## Tree 6 — Image

```text
                    IMAGE TASK
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   Reconstruction      Generation          Detection / Seg
        │                   │                   │
        ▼                   ▼                   ▼
   Reference                Reference?     Detection? → mAP (COCO)
   available?         ┌────┴────┐          Segmentation? → Tree 6a
   ┌────┴────┐        YES        NO
   YES        NO       │          │
    │          │       ▼          ▼
    ▼          ▼      LPIPS      FID + KID
   PSNR       Custom            + CLIPScore
   SSIM      no-ref              + Inception V3 caveat

   Pair-of-images comparison?  →  LPIPS
   Generation prompt fidelity? →  CLIPScore
   Sample diversity?           →  KID + recall (Kynkäänniemi et al.)
```

### Tree 6a — Segmentation

```text
                SEGMENTATION TYPE
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼
         Binary    Multi-class   Boundary?
            │           │           │
            ▼           ▼           ▼
         Dice +     GeneralizedDice  Hausdorff
         IoU        + per-class IoU  (use HD95
                                      for clinical)
                                      ASSD also valid
```

---

## Tree 7 — Audio

```text
                    AUDIO TASK
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   Source separation   Speech enhancement   Speech quality
        │                   │                   │
        ▼                   ▼                   ▼
     SI-SDR              SI-SDR             PESQ (non-diff)
     (differentiable)    + STOI / ESTOI     STOI
                                            DNSMOS
                                            NISQA

   Reverberation?  →  SRMR
   Speaker verification?  →  EER
   Differentiable for training loss?  →  SI-SDR family
   Perceptual quality reporting?      →  PESQ + STOI + DNSMOS
```

---

## Tree 8 — Forecasting / Time series

```text
                FORECASTING TASK
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
       Point forecast?       Probabilistic forecast?
            │                       │
            ▼                       ▼
       wMAPE (cross-series)    CRPS (continuous)
       SMAPE (symmetric)       Quantile loss (asymmetric cost)
       NRMSE (normalized)      Pinball loss (specific quantile)

   Hierarchical (region × store × SKU)?
       →  Per-level wMAPE + reconciliation residual (custom)
   Asymmetric cost (newsvendor)?
       →  Quantile loss at q* = c_under / (c_under + c_over)
   Coverage check?
       →  P(actual ≤ predicted_quantile) — custom metric
```

---

## Tree 9 — Anomaly / fraud

```text
                ANOMALY DETECTION
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
       Supervised?             Unsupervised?
            │                       │
            ▼                       ▼
       Tree 2 + ↓              Score-based + label some
                                       │
                                       ▼
                               Use scores → human label
                               → re-evaluate with Tree 2

   Always include:
   - AP (rare positives)
   - Recall@FixedPrecision (operating point)
   - Per-segment breakdown (catch fairness regressions)
   - Cost-of-error custom metric ($-loss)
```

---

## Tree 10 — Distributed considerations

```text
                YOUR METRIC HAS LIST STATE?
                        │
            ┌───────────┴───────────┐
            YES                     NO
             │                       │
             ▼                       ▼
        Eval set huge?          Tensor state
        ┌────┴────┐              (sum/mean/min/max
        YES        NO              reduces clean)
         │          │
         ▼          ▼
   compute_on    Default
   _cpu=True     OK

   Need step-level synced metric?
       →  dist_sync_on_step=True   (rare; expensive)
   Need rank-only diagnostic?
       →  sync_on_compute=False
   Need persistent metric in checkpoint?
       →  add_state(..., persistent=True)
```

---

## How to use these trees in an interview

1. **Memorize Tree 1 first** (the master flowchart) — you should be able to draw it from memory.
2. The domain trees are reference — you don't need to memorize them, but you should know **the questions** at each branch.
3. When asked "which metric for X," **say the question** before the answer:
   > *"My first question is: is this imbalanced? OK, given the 1 % positive rate, I'm going AP-first. The second question is: what's our hard constraint? You said 99 % precision, so RecallAtFixedPrecision. Now per-segment because…"*

   You're walking the tree out loud. That's senior behavior.

4. **Drill it**. Ask a friend to give you 10 random "which metric" prompts; time yourself. Goal: ≤ 30 s per answer.
