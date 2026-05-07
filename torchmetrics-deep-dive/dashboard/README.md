# TorchMetrics Deep Dive Dashboard

A static, single-page HTML dashboard for revising the TorchMetrics Deep Dive content
and stress-testing yourself with multi-level interview questions.

## How to use

Just open `index.html` in your browser. No build step, no server required.

```bash
# macOS
open index.html

# Linux
xdg-open index.html

# Windows
start index.html
```

Or serve it with any static server if you prefer:

```bash
python -m http.server --directory dashboard 8080
# then visit http://localhost:8080
```

## Modes

- **📖 Revise** — concise topic summaries (formulae, code patterns, decision rules).
- **🎯 Quiz** — interview questions with multi-level follow-ups (Q → F1 → F1.1 → F1.1.1).
  Click *Show answer* to reveal each answer, then click each *Show follow-up* to drill
  one level deeper.

## Topics

Organized into seven categories:

1. **Foundations** — Getting Started, Core Concepts, Metric Class Internals.
2. **Domain Metrics** — Classification, Regression, Retrieval, Text/Audio/Image.
3. **Distributed & Lightning** — DDP, PyTorch Lightning Integration.
4. **Custom & Production** — Custom Metrics, Production Scenarios.
5. **Interview Prep** — Quick-fire, System Design.
6. **Business Mapping** — American Airlines, Amazon, Scenario Setups.
7. **Reference** — Troubleshooting.

## Keyboard shortcuts

- `R` — switch to Revise mode
- `Q` — switch to Quiz mode
- `←` / `→` — previous / next question (in Quiz mode)

## Files

| File | Purpose |
|---|---|
| `index.html` | Page structure |
| `styles.css` | All styling |
| `data.js` | All topics, summaries, and Q&A trees |
| `app.js` | Mode switching, search, quiz state, rendering |

To extend with your own topics, edit `data.js`. The topic schema is documented in
that file's header comment.
