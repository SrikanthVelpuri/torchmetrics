# TorchMetrics Mastery Dashboard

Six modes for revising and stress-testing your TorchMetrics knowledge.

## Open it

```bash
open index.html        # macOS
xdg-open index.html    # Linux
start index.html       # Windows
```

Or any static server:

```bash
python -m http.server --directory dashboard 8080
# visit http://localhost:8080
```

## Modes

### Per-topic modes (top-right toggle)

- **📖 Revise** — concise topic summaries (formulae, code, decision rules).
- **🎯 Quiz** — interview questions with reveal-on-click multi-level follow-ups (Q → F1 → F1.1 → F1.1.1).
- **🃏 Flashcards** — one card at a time, flip to reveal, rate confidence 1–5★.

### Global modes (sidebar)

- **🎲 Random Mix** — random question from anywhere; practice unpredictability.
- **🃏 Flashcards (All)** — weak-first queue across all topics, ordered by lowest confidence + oldest seen.
- **📊 Mastery Map** — heat-map of every question colored by your confidence. Click any cell to jump.

## Confidence ratings (★)

Every question has a 5-star rating. Click any star to commit (click the same star to clear).

- **Persisted to `localStorage`** (key: `tm-confidence`).
- **Drives the weak-first queue** in Flashcards (All).
- **Color-codes the sidebar** (red = weak, green = strong).
- **Resettable** via the sidebar's "↺ Reset progress" button.

## Keyboard shortcuts

- `R` — switch to Revise
- `Q` — switch to Quiz
- `F` — switch to Flashcards
- `Space` — flip card (in Flashcards mode)
- `1` – `5` — rate confidence after flipping
- `←` / `→` — previous / next

## Files

| File | Purpose |
|---|---|
| `index.html` | Page structure |
| `styles.css` | Clean modern UI |
| `data.js` | All topics, summaries, and Q&A trees |
| `app.js` | Mode switching, search, quiz/flash state, confidence persistence |

## Suggested daily ritual

```
25 min   Revise (one topic per day from the roadmap)
10 min   Random Mix (5–10 random cards)
 5 min   Flashcards (All) — drill the weak-first queue
```

After 30 days at this pace, the Mastery Map should be mostly green. That's interview-ready.
