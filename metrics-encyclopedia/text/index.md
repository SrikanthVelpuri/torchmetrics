---
title: Text / NLP metrics — deep dive
---

# Text / NLP metrics — deep dive

> Text metrics fall into four groups:
> - **n-gram overlap** (BLEU, SacreBLEU, ROUGE, chrF) — count token overlap.
> - **edit-based** (TER, EED, edit distance, CER, WER, MER, WIL, WIP) — count operations to transform prediction → reference.
> - **embedding-based** (BERTScore, InfoLM) — compare contextual embeddings.
> - **language-model-based** (perplexity, SQuAD F1/EM) — task-specific.

[← Home](../index.md) · [Interview deep-dive →](./interview-deep-dive.md)

## n-gram overlap family

### BLEU (`tm.text.BLEUScore`)

**What it computes.** Precision over n-grams (n=1..4 typically), with a brevity penalty: `BP · exp(Σ w_n log p_n)`.

**Intuition.** "What fraction of the predicted n-grams appear in the reference?" The BP penalises predictions shorter than the reference (a model that outputs single words has perfect 1-gram precision but is useless).

**Range / direction.** `[0, 1]`. Higher better. SOTA MT systems hit ~0.40 BLEU on WMT.

**When to use.** Machine translation. The historical de-facto metric.

**When NOT to use.** Short texts (n-gram counts are noisy at length < 20). Open-ended generation (paraphrases differ in wording but mean the same thing).

**Real-world scenario.** WMT translation leaderboard — every paper reports BLEU. Internally, MT teams gate launches on BLEU + human evaluation.

**Code.**
```python
from torchmetrics.text import BLEUScore
bleu = BLEUScore(n_gram=4, smooth=True)
bleu([prediction], [[reference1, reference2]])  # multiple refs allowed
```

**Pitfalls.**
- **Tokenization-dependent.** BLEU on different tokenisers gives different numbers. Always use SacreBLEU for cross-paper comparison.
- Brevity penalty is *over the corpus*, not per-sentence — corpus BLEU ≠ mean of sentence BLEU.

---

### SacreBLEU (`tm.text.SacreBLEUScore`)

**What it computes.** BLEU with a *standardised* tokeniser, reproducible across implementations.

**Real-world scenario.** Cross-paper BLEU reporting. Modern WMT requires SacreBLEU. Always use this in publications.

---

### ROUGE (`tm.text.ROUGEScore`)

**What it computes.** Recall over n-grams (and longest-common-subsequence for ROUGE-L).
- **ROUGE-N**: n-gram recall.
- **ROUGE-L**: longest common subsequence based F1.
- **ROUGE-Lsum**: ROUGE-L over multi-sentence outputs.

**Intuition.** BLEU's recall-oriented twin. Reward is for covering *all* the reference n-grams.

**Range / direction.** `[0, 1]`. Higher better.

**When to use.** Summarisation. CNN/DailyMail and XSum leaderboards report ROUGE-1, ROUGE-2, ROUGE-L.

**Pitfalls.**
- ROUGE rewards lexical overlap; abstractive summaries with paraphrasing score lower than extractive ones with identical content. Co-report BERTScore.

---

### chrF (`tm.text.CHRFScore`)

**What it computes.** Character-level n-gram F-score. F-beta with β=1 by default, β=2 (chrF++) emphasises recall.

**Intuition.** Robust to morphology — "cat" vs "cats" share 3-grams that word-BLEU treats as different tokens. Important for morphologically-rich languages.

**Real-world scenario.** WMT for low-resource and morphologically-rich languages: chrF correlates better with human judgments than BLEU on Finnish, Turkish, Russian.

---

## Edit-based family

### EditDistance (`tm.text.EditDistance`)

**What it computes.** Minimum number of insert/delete/substitute operations to transform prediction → reference.

**Real-world scenario.** Spell-correction accuracy: "received 12 keystrokes from intended" — direct interpretation.

---

### TER — Translation Edit Rate (`tm.text.TranslationEditRate`)

**What it computes.** `EditDistance / |reference|`. Words-level edit-based MT metric.

**Range / direction.** `[0, ∞)`. **Lower better.**

**Real-world scenario.** Post-editing effort estimation in MT: TER predicts how many words a human translator will need to fix.

---

### EED — Extended Edit Distance (`tm.text.ExtendedEditDistance`)

**What it computes.** Edit distance with character-level operations and a transposition penalty.

---

### WER, CER, MER, WIL, WIP — Speech / OCR family

These all measure how close a transcription is to truth. Used heavily in ASR and OCR.

- **WER (Word Error Rate)**: `(S + D + I) / N` — substitutions + deletions + insertions over reference word count. The dominant ASR metric. **Lower better.**
- **CER (Character Error Rate)**: same formula at the character level. Used for languages without word boundaries (Chinese, Thai) and for OCR.
- **MER (Match Error Rate)**: `(S + D + I) / (H + S + D + I)` where H = correct words. A normalised WER, bounded to `[0, 1]`.
- **WIL (Word Information Lost)**: information-theoretic ASR error.
- **WIP (Word Information Preserved)**: `1 - WIL`.

**Real-world scenario (WER).** Speech recognition system evaluation. SOTA English ASR ~5% WER on LibriSpeech-clean. Production target: < 8% on the team's domain.

**Pitfalls (WER).**
- Can exceed 1.0 — a transcription with many insertions has more errors than reference words. Don't claim "WER 0-100%."
- Tokenisation matters. "I'm" vs "I am" — same meaning, different WER.

---

## Embedding-based

### BERTScore (`tm.text.BERTScore`)

**What it computes.** Per-token cosine similarity between BERT embeddings of prediction and reference, aggregated to F1.

**Intuition.** Captures *semantic* similarity, not lexical. A paraphrase scores high BERTScore but low BLEU.

**Range / direction.** `[0, 1]`. Higher better. Baselines around 0.85-0.90 — interpret deltas, not absolutes.

**When to use.** Open-ended generation, paraphrase, summarisation. Anywhere wording varies but meaning is what counts.

**When NOT to use.** When literal wording matters (legal, code generation). BERTScore is too lenient there.

**Real-world scenario.** Summarisation eval beyond ROUGE. BERTScore correlates better with human "is this summary on-topic" judgments.

**Pitfalls.**
- Backbone (`roberta-large`, `bert-base`, `microsoft/deberta-xlarge-mnli`) matters — pin the choice.
- Baseline rescaling — BERTScore is naturally in `[0.85, 0.95]`; with `rescale_with_baseline=True` it expands to `[0, 1]`. Pin the choice.

---

### InfoLM (`tm.text.InfoLM`)

**What it computes.** Information-theoretic distance between language-model probability distributions over the prediction and reference.

**Real-world scenario.** Text quality evaluation when you have a strong LM available; correlates well with human judgments.

---

## Language-model-based

### Perplexity (`tm.text.Perplexity`)

**What it computes.** `exp(− (1/N) Σ log p(x_i))` — exponentiated mean negative log-likelihood.

**Intuition.** "How surprised is the model by the text on average?" Lower = better fit.

**Range / direction.** `[1, ∞)`. **Lower better.** WikiText-103 SOTA perplexity ~10; GPT-2-base ~30.

**When to use.** Language-model training tracking. Comparing two LMs on the same dataset and tokeniser.

**When NOT to use.** Comparing models with different vocabularies/tokenisers — perplexity is not comparable across tokenisers.

**Real-world scenario.** Pretraining: perplexity is the headline number. Don't compare across tokenisers; do compare within the same family.

**Pitfalls.**
- **Not comparable across tokenisers**: BPE-32k vocab vs SentencePiece-50k vocab gives different perplexities for identical models. Use bits-per-byte (BPB) for cross-tokeniser comparison.

---

### SQuADScore — SQuAD F1 / EM (`tm.text.SQuAD`)

**What it computes.**
- **EM (Exact Match)**: 1 if normalised predicted answer == one of the gold answers.
- **F1**: token-overlap F1 between predicted and gold answers.

**Real-world scenario.** Extractive question-answering benchmarks. SQuAD-1.1 SOTA: F1 ~93, EM ~88. Standardised normalisation (lowercase, strip punctuation, drop articles) baked in.

---

## Quick-reference: which text metric for which scenario?

| Scenario | Primary | Secondary |
|---|---|---|
| MT (paper, cross-team) | SacreBLEU | chrF |
| MT (low-resource, morph-rich) | chrF / chrF++ | BLEU |
| Summarisation | ROUGE-1/2/L | BERTScore |
| Open-ended generation | BERTScore | InfoLM |
| Question answering (extractive) | SQuAD F1 / EM | — |
| Speech recognition | WER | CER |
| OCR / Chinese / Thai | CER | WER |
| Spell correction | EditDistance | — |
| MT post-editing effort | TER | — |
| LM training | Perplexity | bits-per-byte |
| Code generation | Exact match + execution | — |

---

[Continue → interview deep-dive](./interview-deep-dive.md)
