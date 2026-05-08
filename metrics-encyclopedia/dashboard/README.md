# Dashboard

Self-contained static HTML interactive dashboard for the **Metrics Encyclopedia**.

## Run it

Open [`index.html`](./index.html) directly in a browser:

```bash
# Windows
start metrics-encyclopedia\dashboard\index.html

# macOS
open metrics-encyclopedia/dashboard/index.html

# Linux
xdg-open metrics-encyclopedia/dashboard/index.html
```

No server required. Data lives in [`data.js`](./data.js); the page loads it as a `<script>` tag.

When deployed via GitHub Pages, this lives at `<site>/dashboard/index.html`.

## Five modes

| Mode | Shortcut | What it does |
|---|---|---|
| 📖 Revise | `R` | Per-metric cards: formula, when/not-when, scenario, code, pitfall. |
| 🎯 Quiz | `Q` | Click-to-expand interview Q&A with multi-level follow-ups. |
| 🃏 Flashcards | `F` | Front: metric name. Flip to see definition + scenario. |
| 🏢 Scenarios | `S` | 25 production system metric stacks with traps. |
| 🔍 Search | `/` | Full-text search across all metrics, scenarios, definitions. |

## Keyboard shortcuts

- `R` `Q` `F` `S` — switch modes
- `/` — focus search
- `Space` — flip flashcard
- `←` `→` — previous / next flashcard

## Sidebar

Pick the metric family (left panel). The selected family drives every mode except `Scenarios` (which is cross-family) and `Search` (also cross-family).

## Schema

`data.js` contains two top-level objects:

```js
const METRICS = {
  classification: {
    title: "Classification",
    summary: "...",
    metrics: [{ name, what, when, notWhen?, range, direction: "up"|"down",
                task?, scenario, code?, pitfall? }, ...],
    questions: [{ q, a, followups?: [{q, a}, ...] }, ...]
  },
  ...
};
const SCENARIOS = [{ name, stack, why, trap? }, ...];
```

## Adding a metric

1. Open [`data.js`](./data.js).
2. Add an entry to the relevant family's `metrics` array.
3. Refresh the page.

## Adding an interview question

1. Open [`data.js`](./data.js).
2. Add an entry to the family's `questions` array.
3. Optionally add nested `followups`.

## Adding a scenario

Append to `SCENARIOS` at the bottom of [`data.js`](./data.js).
