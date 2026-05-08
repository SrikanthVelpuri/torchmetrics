# Metrics Encyclopedia

Per-metric deep-dive site for TorchMetrics. Sibling to [`torchmetrics-deep-dive`](../torchmetrics-deep-dive/) — that site teaches the **framework**, this site is the **per-metric encyclopedia**.

## Reading on GitHub

Just click [`index.md`](./index.md) and follow the links. Markdown renders inline.

## Publishing as GitHub Pages

This folder is a self-contained Jekyll site. To publish:

1. Repository **Settings → Pages**.
2. Source: **Deploy from a branch**.
3. Branch: your branch · Folder: `/metrics-encyclopedia`.
4. Save. Site lives at `https://<your-user>.github.io/torchmetrics/`.

If you also want the sibling `torchmetrics-deep-dive` site live, you can only point Pages at one folder — pick one as primary and link to the other from its index.

## Layout

```
metrics-encyclopedia/
  index.md                   ← landing
  _config.yml                ← Jekyll config
  classification/
    index.md                 ← every classification metric
    interview-deep-dive.md   ← Q → F1 → F1.1 drill-downs
  regression/
  retrieval/
  detection/
  segmentation/
  image/
  text/
  audio/
  multimodal/
  video/
  clustering/
  nominal/
  wrappers/
  aggregation/
  scenarios/
    index.md                 ← cross-family system designs
```

Every family page uses the same per-metric template (formula, intuition, range, when to use, when not to use, real-world scenario, code, pitfalls). Every interview page uses the same Q → F1 → F1.1 drill-down format.
