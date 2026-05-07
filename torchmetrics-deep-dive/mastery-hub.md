---
title: 🎓 Mastery Hub
nav_order: 0
---

# 🎓 Mastery Hub — Become Perfect at TorchMetrics

A standalone landing page for **interview-grade mastery** of TorchMetrics. If you only have time to bookmark one page, bookmark this one.

> *"Knowledge is power. Knowledge that's organized so you can use it under interview pressure is **mastery**."*

---

## What this hub gives you

```text
┌─────────────────────────────────────────────────────────────────┐
│  THE MASTERY STACK                                               │
│                                                                   │
│  📋 Roadmap        ──► 30 days, 45 min/day                        │
│  📑 Cheat Sheets   ──► visual one-pagers, scan in 60 sec          │
│  🧠 Mnemonics      ──► remember 100+ metrics with 21 hooks        │
│  🌳 Decision Trees ──► "which metric for X?" in <30 sec           │
│  📦 Wrappers       ──► all 11 wrappers, with composition recipes  │
│  💻 Code Challenges──► implement 12 metrics from scratch          │
│  🐛 Spot the Bug   ──► 15 buggy snippets to debug                 │
│  🔢 Numerical      ──► NaN, precision, edge cases                 │
│  🏢 10 Companies   ──► Netflix, Uber, Stripe, Meta, …             │
│  🎯 Dashboard      ──► flashcards, random mix, mastery map        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Visual: the four pillars of mastery

```
       ┌──────────────────────────────────────────────────┐
       │                  MASTERY                         │
       │                                                  │
       │     CONCEPTS    +    PATTERNS                    │
       │        +              +                          │
       │   IMPLEMENTATION + COMMUNICATION                 │
       │                                                  │
       └──────────────────────────────────────────────────┘
            ▲              ▲              ▲              ▲
            │              │              │              │
            │              │              │              │
       Core Concepts   Cheat Sheets    Code Challenges  ML→Business
       Class Internals Decision Trees   Spot the Bug    Scenarios
                       Mnemonics        Numerical
```

You need all four to "answer any question." Most interview prep stops at concepts. **This site goes all four.**

---

## The 7-tier learning path

### Tier 0 — Foundation (you must have)

You have to be able to write the lifecycle from memory in 60 seconds.

- [Getting Started](./getting-started.md)
- [Core Concepts](./core-concepts.md)
- [Metric Class Internals](./metric-class-internals.md)

> **Self-check**: can you draw the lifecycle (Reset → Update → Update → Compute → Compute) and name the 4 standard `dim_zero_*` reductions without looking? If yes, advance.

### Tier 1 — Domain coverage

Pick the domains you actually work in. Skim the others; you should know they exist and what they're for.

- [Classification](./classification-metrics.md) — task taxonomy, AUROC vs AP, calibration.
- [Regression](./regression-metrics.md) — MAE/MSE/wMAPE, R² gotchas.
- [Retrieval](./retrieval-metrics.md) — `indexes`, NDCG/MRR.
- [Text/Audio/Image](./text-audio-image-metrics.md) — BLEU/FID/PESQ.
- [Aggregation, Clustering, Pairwise, Nominal](./aggregation-clustering-pairwise.md) — *the underdocumented families*.

### Tier 2 — Distributed mastery

This is where mid-level becomes senior.

- [Distributed Training (DDP)](./distributed-training.md)
- [PyTorch Lightning Integration](./pytorch-lightning-integration.md)

> **Self-check**: can you explain `_sync_dist` line by line, including the empty-rank case? If yes, advance.

### Tier 3 — Customization

Real interview gold: implementing your own.

- [Custom Metrics](./custom-metrics.md)
- [Wrappers Deep Dive](./wrappers-deep-dive.md)
- [Code Challenges](./code-challenges.md) — work all 12 cold.

### Tier 4 — Production pragmatics

The "I've actually shipped this" tier.

- [Production Scenarios](./production-scenarios.md)
- [Scenario Setups](./scenario-setups.md)
- [Numerical Pitfalls](./numerical-pitfalls.md)
- [Spot the Bug](./spot-the-bug.md)
- [Testing & Validation](./testing-and-validation.md)
- [Troubleshooting](./troubleshooting.md)

### Tier 5 — Business translation

The "staff engineer" tier — connecting ML to dollars.

- [ML ↔ Business Metrics](./ml-business-metrics.md) (American Airlines + Amazon)
- [Extended Company Scenarios](./extended-company-scenarios.md) (Netflix, Uber, Stripe, Meta, Tesla, healthcare, finance, cybersecurity)

### Tier 6 — Interview combat

Practice under pressure.

- [Interview Questions](./interview-questions.md)
- [Follow-Up Questions](./follow-up-questions.md)
- [System Design Questions](./system-design-questions.md)
- [Dashboard](./dashboard/index.html) (Quiz mode + Flashcards + Random Mix)

---

## Visual clarity tools — by purpose

### To **scan quickly** (review at glance):

- [Cheat Sheets](./cheat-sheets.md) — 18 visual one-pagers covering every major topic. Print these.

### To **decide fast** (when asked "which metric"):

- [Metric Decision Trees](./metric-decision-tree.md) — 10 ASCII flowcharts walking branches.

### To **remember long-term**:

- [Mnemonics](./mnemonics.md) — 21 memory hooks (RUUCC, MEPP, PRINCE, SUMMINCAT, …).

### To **practice actively**:

- [Dashboard](./dashboard/index.html) — Quiz / Flashcards / Random Mix modes with confidence-rating spaced repetition.

---

## "Help me remember" — the memory stack

Memory works in layers. Most students try to memorize at the wrong layer.

```
LAYER                           WHERE YOU'D PRACTICE IT
─────────────────────────────────────────────────────────────────
RECOGNITION (passive)           Reading the docs
RECALL (active)                 Quiz mode (cover-and-answer)
CHUNKING (compressed)           Mnemonics (RUUCC, MEPP, PRINCE)
CONNECTING (associative)        Decision trees ("if X then…")
SYNTHESIS (productive)          Code challenges, spot-the-bug
ARTICULATION (verbal)           ML→Business pitch (out loud)
```

The dashboard's **Spaced Repetition** is the active-recall + chunking + connecting loop. If you only do one thing daily for 30 days, do that.

---

## Memorization techniques specific to TorchMetrics

### 1. **The 5-3-2 rule** (numerical anchors)

```
5  lifecycle methods       (init, update, forward, compute, reset)
4  stat scores             (TP, FP, TN, FN)
3  classification tasks    (binary, multiclass, multilabel)
2  forward modes           (full state vs reduce state update)
```

Add **1** for `MetricCollection`. Now you have **5-4-3-2-1** — a launch countdown that anchors all of TorchMetrics.

### 2. **The metric MUSEUM** (method of loci)

Walk through your imagined museum every morning. 5 rooms, ~20 metrics per room.

Visit [Mnemonics](./mnemonics.md) → "The metric MUSEUM" for the layout.

### 3. **Pattern-matching cards**

Make 30 index cards. Front: scenario ("imbalanced fraud, 99 % precision constraint"). Back: metric ("Recall@FixedPrecision(min_precision=0.99) + AP + ECE"). Cycle through daily.

### 4. **Story-driven learning**

When learning a metric, attach it to a **user story**:

- AUROC ↔ "the radiologist comparing models on a leaderboard"
- NDCG ↔ "the SRE tuning Netflix homepage"
- VMAF ↔ "the encoding engineer fighting bandwidth budgets"
- Pinball loss ↔ "the inventory planner avoiding stockouts"

A metric attached to a person is 5× more memorable than a formula in isolation.

### 5. **Spaced repetition (the science)**

```
First exposure → review SAME DAY before sleep    (consolidation)
Day 2          → review once                      (early reinforcement)
Day 4          → review once                      (medium-term)
Day 8          → review once                      (long-term)
Day 16         → review once                      (mastery)
Day 32         → review once                      (lifelong)
```

The dashboard has built-in SR with star confidence. Use it.

---

## "Answer any question" — the breadth checklist

Tick these off. When all are ✅, you're ready.

### Conceptual (Tier 0–1)

- [ ] Explain TorchMetrics' purpose to a non-engineer in 60 sec.
- [ ] Recite the lifecycle from memory.
- [ ] Distinguish functional vs modular API with an example.
- [ ] Explain why metrics are non-decomposable, with a numerical example.
- [ ] Walk through `forward()` — both fast and safe paths.
- [ ] Describe `_sync_dist` end-to-end.

### Domain (Tier 1)

- [ ] Pick the right metric for any classification task in <30 sec.
- [ ] Explain when AUROC fails and AP wins.
- [ ] Justify wMAPE vs MAPE for a low-volume forecasting series.
- [ ] Build a recommender retrieval stack (recall@K + reranker NDCG).
- [ ] Explain FID's small-N bias and the KID alternative.

### Engineering (Tier 2–3)

- [ ] Implement `WeightedMean` from memory in 5 min.
- [ ] Implement `PinballLoss` (multi-quantile) from memory in 15 min.
- [ ] Implement `DollarLoss` from memory in 10 min.
- [ ] Find at least 12/15 bugs in [Spot the Bug](./spot-the-bug.md).
- [ ] Pick the right wrapper for any "I want X but with Y" prompt.

### Production (Tier 4)

- [ ] List 4 numerical pitfalls.
- [ ] Diagnose a DDP-broken metric from a stack trace.
- [ ] Design a rolling-window monitoring metric for a 50k-QPS recommender.
- [ ] Reproduce sklearn's AUROC to 1e-6.

### Business (Tier 5)

- [ ] Bridge any ML metric to a business KPI with the **CALM** framework.
- [ ] Explain newsvendor / quantile cost asymmetry.
- [ ] Pick metrics for *5 different companies* without looking.
- [ ] Defend a launch decision under metric ambiguity.

### Interview combat (Tier 6)

- [ ] Drill 4 levels deep on at least 3 topics.
- [ ] Pass a 60-min mock interview (Day 30).
- [ ] Articulate the **STAR-M** pitch for any metric question.

---

## "I have only N minutes before the interview" — emergency cards

| Time | Read this |
|---|---|
| 5 min | [Cheat Sheets](./cheat-sheets.md) — Lifecycle + Classification chooser + Imbalance kit |
| 15 min | + [Mnemonics](./mnemonics.md) — RUUCC, MEPP, PRINCE, CALM |
| 30 min | + [Metric Decision Trees](./metric-decision-tree.md) Tree 1 + the domain you'll be asked about |
| 1 hour | + [Spot the Bug](./spot-the-bug.md) (skim) + [Numerical Pitfalls](./numerical-pitfalls.md) (skim) |
| Half day | Quiz mode in [Dashboard](./dashboard/index.html) on Random Mix |
| Full day | Day 28 + Day 30 mock interviews from the [Roadmap](./mastery-roadmap.md) |

---

## "I'm an interviewer; what should I ask?"

Sample prompts at three levels.

### Junior level (graduating ICs)

- "What's the difference between a metric's `update` and `compute`?"
- "When would you choose Accuracy over F1?"
- "Why do we need `reset()`?"

### Mid level

- "Walk me through what happens when I call `metric(preds, target)` in DDP."
- "Implement Weighted MAE in 10 min."
- "Why isn't softmax 'good enough' for calibration?"

### Senior / staff

- "Design the metric layer of a 50k-QPS recommender with rolling-window monitoring."
- "Your offline NDCG improves by 0.5 % but online lift is flat. What's happening?"
- "How would you measure fairness across an intersectional protected attribute?"
- *(then) drill 3 levels deeper into whichever they answer.*

---

## Bottom line

You don't memorize 100+ metrics. You memorize **patterns**:

- **5-4-3-2-1** anchor for the API.
- **CALM** for ML→business.
- **MEPP** for classification picks.
- **PRINCE** for wrapper picks.
- **SIDS** for DDP correctness.
- **NIPS** for numerical traps.
- **STAR-M** for the verbal answer template.

Patterns compress information. Once compressed, recall is fast even under pressure. That's mastery.

---

## What to bookmark

- 📌 This page (Mastery Hub) — the index.
- 📌 [Cheat Sheets](./cheat-sheets.md) — the daily 5-min review.
- 📌 [Decision Trees](./metric-decision-tree.md) — for "which metric" questions.
- 📌 [Dashboard](./dashboard/index.html) — for daily practice.

Three pages + the dashboard. That's your mastery loop.
