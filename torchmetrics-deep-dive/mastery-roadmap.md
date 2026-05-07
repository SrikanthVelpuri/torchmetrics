---
title: Mastery Roadmap
nav_order: 20
---

# Mastery Roadmap — 30 Days to TorchMetrics Fluency

A structured, time-boxed plan that takes you from "I've heard of TorchMetrics" to "I can answer any TorchMetrics question." Each day is **45–60 min**. Treat it like a workout split — same time slot, every day.

> **The honest truth.** You won't memorize 100+ metrics. You'll memorize the **patterns** (5 lifecycle methods, 4 stat scores, 3 reduction types, 2 forward modes). Everything else clicks once those are wired in.

---

## Visual progress map

```text
WEEK 1 — MENTAL MODEL                 WEEK 3 — DEPTH
█ Day  1  Why TM exists               █ Day 15  Internals line-by-line
█ Day  2  Functional vs modular       █ Day 16  _sync_dist & DDP
█ Day  3  add_state / update / compute █ Day 17  Wrappers (all 11)
█ Day  4  reset & lifecycle            █ Day 18  Custom metric: build one
█ Day  5  MetricCollection             █ Day 19  Custom metric: DDP-test it
█ Day  6  Quiz day (review week 1)     █ Day 20  Code challenges (5)
█ Day  7  Visualize: lifecycle diagram █ Day 21  Quiz day (review week 3)

WEEK 2 — DOMAINS                       WEEK 4 — MASTERY
█ Day  8  Classification               █ Day 22  Spot-the-bug exercises
█ Day  9  Regression                   █ Day 23  Numerical pitfalls
█ Day 10  Retrieval                    █ Day 24  ML→business: AA
█ Day 11  Text/Audio/Image             █ Day 25  ML→business: Amazon
█ Day 12  Detection / Segmentation     █ Day 26  ML→business: extended (Netflix, Uber, …)
█ Day 13  Clustering / Pairwise        █ Day 27  System design (recsys, FID)
█ Day 14  Quiz day (review week 2)     █ Day 28  Mock interview (45min, mixed)
                                       █ Day 29  Cheat-sheet recall test
                                       █ Day 30  Final mock (60min, harder)
```

---

## Daily structure (the 45-minute block)

```
┌──────── 5 min ────────┬──────── 25 min ────────┬──────── 10 min ────────┬── 5 min ──┐
│  RECALL yesterday     │  STUDY today's topic   │  PRACTICE quiz mode    │  WRITE 3  │
│  (no reading,         │  (read + take notes,   │  (random questions     │  things   │
│   write key points)   │   build mental model)  │   from the dashboard)  │  learned  │
└───────────────────────┴────────────────────────┴────────────────────────┴───────────┘
```

The 5-min RECALL at the start is the most important block. **Active recall beats re-reading by 2–3×** for retention.

---

## Week 1 — Build the mental model

### Day 1 — Why TorchMetrics exists

- 📖 Read: [Index](./index.md), [Getting Started](./getting-started.md).
- 🎯 Quiz: 3 questions on Getting Started.
- ✍️ Write: "What does TorchMetrics solve that I can't easily do myself?" (3 sentences.)

### Day 2 — Functional vs modular

- 📖 Read: [Getting Started](./getting-started.md) — both APIs.
- 💻 Code: Write a 30-line script that uses each API on the same data and asserts they agree.
- 🎯 Quiz: Functional vs modular question (Q1 in Getting Started).

### Day 3 — `add_state`, `update`, `compute`

- 📖 Read: [Core Concepts](./core-concepts.md) — first half.
- ✍️ Draw the lifecycle diagram from memory.
- 🎯 Quiz: 3 questions on Core Concepts.

### Day 4 — `reset` and the full lifecycle

- 📖 Read: [Core Concepts](./core-concepts.md) — second half.
- 💻 Code: Write a tiny custom `MeanMetric` from scratch (no copy-paste).
- 🎯 Quiz: lifecycle questions.

### Day 5 — `MetricCollection` and compute groups

- 📖 Read: [Core Concepts](./core-concepts.md) — section "MetricCollection and compute groups".
- 💻 Code: Build a `MetricCollection` of 4 classification metrics; verify the speedup vs running each separately.
- ✍️ Mnemonic: invent your own for the 5 lifecycle methods. (Spoiler: see [Mnemonics](./mnemonics.md).)

### Day 6 — Quiz day

- 🎯 Run all of Week 1 in Quiz mode. Star (★) anything you missed.
- ✍️ Write a 1-page summary of Week 1 from memory. Compare to source.

### Day 7 — Lifecycle diagram

- 🎨 Draw the lifecycle on a whiteboard / paper. Explain it out loud as if teaching.
- 📖 Read: [Cheat Sheets](./cheat-sheets.md) — lifecycle section.
- ✍️ Three things you didn't fully understand. Look them up.

---

## Week 2 — Domain coverage

### Day 8 — Classification

- 📖 Read: [Classification](./classification-metrics.md) (incl. drill-down).
- 🎯 Quiz: classification.
- ✍️ Decision tree: "Which classification metric for X?" (compare to [Decision Tree](./metric-decision-tree.md)).

### Day 9 — Regression

- 📖 Read: [Regression](./regression-metrics.md).
- 🎯 Quiz: regression + drill-down.
- 💻 Code: implement `WeightedMAE` from scratch.

### Day 10 — Retrieval

- 📖 Read: [Retrieval](./retrieval-metrics.md).
- 🎯 Quiz: retrieval + drill-down.
- ✍️ One paragraph: why does retrieval need `indexes=`?

### Day 11 — Text / Audio / Image

- 📖 Read: [Text/Audio/Image](./text-audio-image-metrics.md).
- 🎯 Quiz: text + image questions.
- ✍️ Distinguish: "I'd use FID/KID/LPIPS/CLIPScore for…" (one line each).

### Day 12 — Detection / Segmentation

- 📖 Read: [Scenario Setups](./scenario-setups.md) — detection + segmentation sections.
- 🎯 Quiz: scenario setups (detection, segmentation).
- ✍️ Why does mAP run on CPU?

### Day 13 — Clustering / Pairwise / Aggregation

- 📖 Read: [Aggregation, Clustering, Pairwise](./aggregation-clustering-pairwise.md).
- 🎯 Quiz: those families.

### Day 14 — Quiz day

- 🎯 All of Week 2 in random-mix mode.
- ✍️ Domain selection cheat sheet from memory.

---

## Week 3 — Depth

### Day 15 — Internals line-by-line

- 📖 Read: [Metric Class Internals](./metric-class-internals.md). Pull up `metric.py` in another tab.
- ✍️ Draw the call stack of `metric(preds, target)` end to end.

### Day 16 — DDP

- 📖 Read: [Distributed Training](./distributed-training.md).
- 🎯 Quiz: DDP drill-down (4 levels deep — go all the way).
- ✍️ Diagram: `_sync_dist` flow.

### Day 17 — Wrappers (all of them)

- 📖 Read: [Wrappers Deep Dive](./wrappers-deep-dive.md).
- ✍️ One sentence per wrapper saying when you'd reach for it.

### Day 18 — Build a custom metric

- 💻 Pick any metric not in TorchMetrics (e.g. **Geometric Mean of Recall**, **Macro Average Precision at k**, **Pairwise t-test p-value tracker**). Implement it.
- ✅ Make sure: tensor states + dist_reduce_fx, `is_differentiable` set, doctests pass.

### Day 19 — DDP-test the custom metric

- 💻 Write a `world_size=2` test that asserts your metric matches the single-process result.
- ✍️ Document any gotchas you hit.

### Day 20 — Code challenges

- 💻 [Code Challenges](./code-challenges.md) — solve 5 cold (no copy).
- ✍️ For each, what was the gotcha?

### Day 21 — Quiz day

- 🎯 Random-mix across all of Weeks 1–3.

---

## Week 4 — Mastery

### Day 22 — Spot-the-bug

- 🐛 [Spot the Bug](./spot-the-bug.md) — work through all of them.
- ✍️ Tally: which kind of bug did you miss most often? (Probably DDP-related.)

### Day 23 — Numerical pitfalls

- 📖 [Numerical Pitfalls](./numerical-pitfalls.md).
- ✍️ Three pitfalls you didn't know about.

### Day 24 — ML→business: American Airlines

- 📖 Read: [ML ↔ Business](./ml-business-metrics.md) — AA section only.
- 🎯 Quiz: AA drill-downs (full depth).
- ✍️ Pitch deck: "How would I evaluate the new delay-prediction model to AA's revenue management VP?" 1 slide, 3 bullets.

### Day 25 — ML→business: Amazon

- 📖 Same page, Amazon section.
- 🎯 Quiz: Amazon drill-downs.
- ✍️ Same pitch-deck exercise for an Amazon scenario.

### Day 26 — ML→business: extended

- 📖 [Extended Company Scenarios](./extended-company-scenarios.md).
- ✍️ Pick one that interests you most. Write the metric stack from memory.

### Day 27 — System design

- 📖 [System Design Questions](./system-design-questions.md).
- ✍️ Pick one prompt. Whiteboard it (45 min, on paper, no notes).

### Day 28 — Mock interview I

- 🎤 45 minutes. Mix of: 5 conceptual, 1 code-from-scratch, 1 system design, 1 ML→business. Time yourself.
- ✍️ Score yourself out of 10. What was the weakest area?

### Day 29 — Cheat-sheet recall

- ✍️ Reproduce from memory: lifecycle, classification decision tree, MAE/MSE/RMSE table, NDCG formula, FID formula. Compare to [Cheat Sheets](./cheat-sheets.md).

### Day 30 — Mock interview II

- 🎤 60 minutes. Harder: system design + cost-aware scenario + 1 ambiguous "design a metric" prompt.
- 🏁 Done.

---

## After 30 days — maintenance

Spend ~15 min every other day on Quiz mode in **Random Mix** to keep skills fresh. Add new metrics as the library evolves.

---

## How to use this site for the roadmap

1. Bookmark the [Dashboard](./dashboard/index.html). Open it daily.
2. The dashboard has **🃏 Flashcards**, **🎲 Random Mix**, and **📊 Mastery Map** — use them after Day 14.
3. Track confidence (★1–5) on each question. Surface low-confidence questions first.
4. The site's [Cheat Sheets](./cheat-sheets.md) and [Mnemonics](./mnemonics.md) are designed for daily 5-min recall.

---

## Self-assessment rubric

After Day 30, you should be able to:

- [ ] Explain the **lifecycle** of a `Metric` to a non-technical PM in 60 seconds.
- [ ] Name **all four `dim_zero_*` reductions** without looking.
- [ ] Implement a **custom metric** (with DDP correctness) in under 15 minutes.
- [ ] Pick the right metric for **any of 20+ scenarios** and explain the trade-off.
- [ ] Diagnose a **DDP-broken metric** from a 5-line stack trace.
- [ ] Bridge **any ML metric to a business KPI** with the 4-bridge framework.
- [ ] Answer **4 levels deep** of follow-up on at least 5 topics.

If 7/7 — you're ready. If 5/7 — re-do the weak weeks. Below 5 — start over from Week 1; you went too fast.
