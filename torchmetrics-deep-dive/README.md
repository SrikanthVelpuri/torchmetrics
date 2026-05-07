# TorchMetrics Deep Dive

> A complete interview-prep & mastery site for **TorchMetrics** — the official metrics library for PyTorch / PyTorch Lightning.

📚 **30 markdown pages** · 🎯 **interactive HTML dashboard** · 🧠 **21 mnemonics** · 💻 **12 code challenges** · 🐛 **15 debug exercises** · 🏢 **10 company scenarios**

---

## 🚀 Quick start — 3 ways to view this content

### Option A — Read on GitHub (zero setup)

GitHub renders Markdown automatically. Click any link in the table below — done.

> ✅ Works immediately · ❌ Dashboard (HTML) won't render this way

### Option B — Open the dashboard locally (recommended for the interactive parts)

```bash
git clone https://github.com/Lightning-AI/torchmetrics.git
cd torchmetrics
git checkout claude/torchmetrics-github-pages-nRMK8     # this branch
open torchmetrics-deep-dive/dashboard/index.html        # macOS
# or:  xdg-open torchmetrics-deep-dive/dashboard/index.html  (Linux)
# or:  start torchmetrics-deep-dive/dashboard/index.html    (Windows)
```

> ✅ Everything works (markdown + interactive dashboard) · ❌ Needs a clone

### Option C — Publish via GitHub Pages (best for sharing)

1. Push this branch to your fork.
2. Repository **Settings → Pages**.
3. Source: **Deploy from a branch**.
4. Branch: `claude/torchmetrics-github-pages-nRMK8` · Folder: `/torchmetrics-deep-dive`.
5. Save. Your site lives at `https://<your-user>.github.io/torchmetrics/`.

> ✅ Public URL · ✅ Dashboard renders · ⏳ Takes 1–2 min on first deploy

---

## 🎯 Start here

If you only have time for one page, open one of these:

- 📚 **[All Pages — Site Map](./all-pages.md)** — every URL on this site in one navigable directory.
- 🎓 **[Mastery Hub](./mastery-hub.md)** — the curated 7-tier learning path with bookmark recommendations.

---

## 🗺️ Full site map

### Mastery & memory

| Page | What's in it |
|---|---|
| [🎓 Mastery Hub](./mastery-hub.md) | The master entry. 7-tier path, breadth checklist, emergency timing tables. |
| [📅 Mastery Roadmap](./mastery-roadmap.md) | 30 days, 45 min/day, structured study plan. |
| [📑 Cheat Sheets](./cheat-sheets.md) | 18 visual one-pagers — print and pin. |
| [🧠 Mnemonics](./mnemonics.md) | 21 memory hooks (RUUCC, MEPP, PRINCE, SUMMINCAT, CALM, …). |
| [🌳 Decision Trees](./metric-decision-tree.md) | "Which metric for X?" — 10 ASCII flowcharts. |

### Foundations

| Page | What's in it |
|---|---|
| [Index / Home](./index.md) | Site overview and how to read it. |
| [Getting Started](./getting-started.md) | Install, first metric, functional vs modular. |
| [Core Concepts](./core-concepts.md) | Lifecycle, state types, performance flags. |
| [Metric Class Internals](./metric-class-internals.md) | Line-by-line tour of `metric.py`. |

### Domain metrics

| Page | What's in it |
|---|---|
| [Classification](./classification-metrics.md) | Task taxonomy, AUROC vs AP, calibration, fairness. |
| [Regression](./regression-metrics.md) | MAE/MSE/wMAPE, R² gotchas, correlations. |
| [Retrieval](./retrieval-metrics.md) | NDCG/MRR/MAP, the `indexes=` argument. |
| [Text / Audio / Image](./text-audio-image-metrics.md) | BLEU, ROUGE, BERTScore, FID, SSIM, PESQ. |
| [Aggregation, Clustering, Pairwise, Nominal](./aggregation-clustering-pairwise.md) | The under-documented families. |

### Distributed & Lightning

| Page | What's in it |
|---|---|
| [Distributed Training (DDP)](./distributed-training.md) | `_sync_dist`, `all_gather`, ragged tensors. |
| [PyTorch Lightning Integration](./pytorch-lightning-integration.md) | `self.log`, `on_step` vs `on_epoch`, `clone()`. |

### Custom & wrappers

| Page | What's in it |
|---|---|
| [Custom Metrics](./custom-metrics.md) | The 5-step recipe. |
| [Wrappers Deep Dive](./wrappers-deep-dive.md) | All 11 wrappers + composition recipes. |

### Practice

| Page | What's in it |
|---|---|
| [💻 Code Challenges](./code-challenges.md) | 12 implement-from-scratch problems. |
| [🐛 Spot the Bug](./spot-the-bug.md) | 15 buggy snippets — debug, then check. |
| [🔢 Numerical Pitfalls](./numerical-pitfalls.md) | NaN, precision, FID small-N bias, edge cases. |
| [Testing & Validation](./testing-and-validation.md) | How TorchMetrics tests itself; how to test yours. |

### Production

| Page | What's in it |
|---|---|
| [Production Scenarios](./production-scenarios.md) | Drift, A/B, fairness CI, segmented metrics. |
| [Scenario Setups](./scenario-setups.md) | 10 ML-system metric configurations (ImageNet, detection, ASR, recsys, …). |
| [Troubleshooting](./troubleshooting.md) | Common bugs and how to fix them. |

### Interview prep

| Page | What's in it |
|---|---|
| [Interview Questions](./interview-questions.md) | 25 questions with model answers. |
| [Follow-Up Questions](./follow-up-questions.md) | 25 senior-level drill-downs. |
| [System Design Questions](./system-design-questions.md) | 8 system-design prompts with worked solutions. |

### Business mapping

| Page | What's in it |
|---|---|
| [🏢 ML ↔ Business Metrics](./ml-business-metrics.md) | American Airlines + Amazon scenarios. |
| [🏢 Extended Company Scenarios](./extended-company-scenarios.md) | Netflix, Uber, Stripe, Meta, Google, Tesla, healthcare, finance, cybersecurity, robotics. |

---

## 🎯 Interactive Dashboard

[**`dashboard/index.html`**](./dashboard/index.html) — a self-contained static HTML app. **Six modes:**

| Mode | What it does |
|---|---|
| 📖 **Revise** | Concise summaries of every topic. |
| 🎯 **Quiz** | Multi-level follow-ups: Q → F1 → F1.1 → F1.1.1. |
| 🃏 **Flashcards** | Flip-cards with 1–5★ confidence rating. |
| 🎲 **Random Mix** | Random question from anywhere. |
| 🃏 **Flashcards (All)** | Weak-first spaced-repetition queue. |
| 📊 **Mastery Map** | Heat-map of confidence across every question. |

Confidence ratings persist to `localStorage`. Sidebar pips color-code your weak topics.

**Keyboard shortcuts**: `R`/`Q`/`F` mode-switch · `Space` flip card · `1`–`5` rate · `←`/`→` navigate.

See [`dashboard/README.md`](./dashboard/README.md) for full docs.

---

## ⏱️ Time-budget guide

| You have | Read these (in order) |
|---|---|
| 5 min | [Cheat Sheets](./cheat-sheets.md) — Lifecycle + Classification chooser + Imbalance kit |
| 15 min | + [Mnemonics](./mnemonics.md) — RUUCC, MEPP, PRINCE, CALM |
| 30 min | + [Decision Trees](./metric-decision-tree.md) Tree 1 + your domain |
| 1 hour | + [Spot the Bug](./spot-the-bug.md) (skim) + [Numerical Pitfalls](./numerical-pitfalls.md) (skim) |
| Half day | Quiz mode in the [Dashboard](./dashboard/index.html) on Random Mix |
| Full day | Day 28 + Day 30 mock interviews from the [Roadmap](./mastery-roadmap.md) |
| 30 days | The full [Mastery Roadmap](./mastery-roadmap.md) |

---

## 📊 Stats

- **30 Markdown pages** (~250 KB of content)
- **80+ multi-level interview drill-downs** (Q → F1 → F1.1 → F1.1.1)
- **12 code challenges** with model solutions
- **15 debug exercises**
- **21 mnemonics** for retention
- **18 cheat-sheet visualizations**
- **10 decision trees** for metric selection
- **10 company scenarios** (AA, Amazon, Netflix, Uber, Stripe, Meta, Google, Tesla, healthcare, finance)
- **Interactive dashboard** with 6 modes + spaced repetition

---

## 🤝 Contributing / extending

The site is plain Markdown + Jekyll, with a separate static HTML dashboard. To add content:

- **New topic** → add `<topic>.md` with Jekyll front-matter; link from [`mastery-hub.md`](./mastery-hub.md) and update [`_config.yml`](./_config.yml) navigation.
- **New dashboard topic** → add an entry to `dashboard/data.js` (schema documented at the top of that file).
- **Re-skin the dashboard** → edit `dashboard/styles.css`. The CSS uses CSS custom properties for easy theming.

---

## License

Same as parent repository (Apache-2.0).
