---
title: Text / NLP metrics — interview deep dive
---

# Text / NLP metrics — interview deep dive

[← Family page](./index.md) · [← Home](../index.md)

---

## Q1. "BLEU vs SacreBLEU — why prefer SacreBLEU?"

**Answer.** Plain BLEU is tokeniser-dependent: tokenise differently, get different numbers. SacreBLEU pins the tokeniser (intl, 13a, etc.) so two researchers running on the same data get identical numbers. Modern WMT mandates SacreBLEU. The only reason to use plain BLEU is reproducing very old papers that pre-date SacreBLEU.

> **F1.1** "What's a typical BLEU difference from tokenisation alone?"
>
> **Answer.** 1-3 BLEU points across tokenisers — enough to flip a leaderboard ranking. The interview signal: candidates who know SacreBLEU are paying attention to reproducibility.

---

## Q2. "Why is BLEU bad for short outputs?"

**Answer.** BLEU averages 1- through 4-gram precision (geometric mean). With a 5-word prediction, you have at most 2 four-grams; one bad 4-gram hammers the geometric mean. Plus, brevity penalty plays asymmetric tricks at very short lengths. Below ~15 tokens, BLEU is too noisy; use chrF or BERTScore.

> **F2.1** "What does the smoothing in BLEU do?"
>
> **Answer.** When a sentence has zero matches at a particular n-gram order, the geometric mean is zero — *one bad order zeroes the score*. Smoothing adds a small floor (epsilon) so a single missed 4-gram doesn't nuke the score. Different smoothing methods exist (NIST, BLEU+1, exp); SacreBLEU uses one consistent default.

---

## Q3. "Summarisation evaluation — ROUGE alone enough?"

**Answer.** No. ROUGE rewards lexical overlap. Two summaries with identical meaning but different phrasing ("the team won the match" vs "the squad triumphed in the bout") get very different ROUGE. Co-report **BERTScore** for semantic similarity and have humans rank a sample for the gold-standard.

> **F3.1** "What's a typical correlation between ROUGE and human judgments on summaries?"
>
> **Answer.** Pearson around 0.3-0.5 depending on dataset. Modest. Better correlation: BERTScore (0.5-0.7), QA-based metrics like QuestEval (0.6-0.8). The interview answer: ROUGE is necessary for legacy comparison, not sufficient for ship/no-ship.

---

## Q4. "WER can exceed 100%. Why?"

**Answer.** WER = `(S + D + I) / N` where `N` = reference word count. If the system inserts more words than the reference contains, `I > N` is possible — total errors exceed total reference words. WER stays valid (still a non-negative real) but is not bounded by 1.0. Don't claim "WER is in [0, 100%]"; it isn't. Use **MER** if you need a bounded metric (`(S+D+I) / (H+S+D+I)`).

> **F4.1** "When does WER trip up production teams?"
>
> **Answer.** Hallucination. A model that hallucinates extra words has high `I`, sometimes pushing WER > 1.0 on short utterances. The dashboard then shows "WER = 105%" and stakeholders ask whether the metric is broken. It isn't — your model is.

---

## Q5. "Why is perplexity not comparable across tokenisers?"

**Answer.** Perplexity = `exp(NLL)` where NLL is averaged *per token*. Different tokenisers split text into different numbers of tokens — a BPE-32k vocab might have ~1.3 tokens/word, a 10k vocab ~1.5. So the same model evaluated on the same text has different per-token NLL. Use **bits-per-byte (BPB)** or **bits-per-character (BPC)** for cross-tokeniser comparison — those normalise by raw bytes/characters, which is tokeniser-independent.

> **F5.1** "Convert perplexity = 30 with vocabulary size 50k to bits-per-byte."
>
> **Answer.** Bits per token = `log2(perplexity) = log2(30) ≈ 4.9`. Then `BPB = bits_per_token × tokens_per_byte`. If your tokeniser averages 0.25 tokens/byte (4 bytes/token average), BPB ≈ 4.9 × 0.25 = 1.23. The conversion needs the tokens-per-byte ratio, which depends on language and tokeniser.

---

## Q6. "BERTScore is at 0.91 — is that good?"

**Answer.** Without rescaling, BERTScore is naturally in `[0.85, 0.95]` even for moderate similarity — *don't read raw values, read deltas*. With `rescale_with_baseline=True`, the metric is rescaled relative to a corpus baseline so 0 = chance and 1 = perfect; that's the readable form. Always pin `rescale_with_baseline` and the backbone in your reports.

---

## Q7. "Build a metric stack for an LLM evaluating on multiple tasks."

**Answer.** Three layers:
1. **Task-specific exact metrics**: SQuAD-F1 for QA; pass-rate for code; ROUGE for summarisation.
2. **Semantic checks**: BERTScore on tasks where wording varies.
3. **Language-model checks**: perplexity on held-out same-domain text (training-domain match).
4. **Human eval**: pairwise rankings on a sample of 200, every release.

Plus calibration check on per-token confidence vs accuracy if the model emits probabilities.

---

## Q8. "MT — BLEU went up 1.5, human eval is unchanged. Ship?"

**Answer.** No. BLEU is a *correlate* of human quality, not a replacement. Three ways this can happen:
1. The 1.5 BLEU is from a single test set quirk (specific n-grams better matched).
2. Lexical fidelity went up but fluency/adequacy didn't.
3. The model is gaming a known BLEU pattern.

Always ship on human eval (or a strong proxy like COMET). BLEU is for tracking, not gating.

---

## Q9. "Difference between corpus BLEU and sentence BLEU?"

**Answer.** Corpus BLEU pools n-gram counts across all sentences then computes precision and brevity penalty once. Sentence BLEU computes BLEU per sentence, often averaged. They differ because brevity penalty is non-linear in length — corpus BLEU is the canonical version; mean(sentence BLEU) is biased high. Always specify which you're reporting.

---

## Q10. "What's wrong with reporting only BLEU on a Japanese MT system?"

**Answer.** Japanese has no spaces between words; tokenisation is non-trivial. BLEU on Japanese requires committing to a segmenter (MeCab, Sudachi). Numbers don't compare across segmenters. Use **chrF** which works at character level and avoids segmentation entirely. The interview signal: candidates who default to BLEU regardless of language don't understand its limitations.

---

## Q11. "Speech recognition — WER 8% on LibriSpeech-clean. Production-ready?"

**Answer.** Maybe. LibriSpeech-clean is read audio in studio conditions. Production audio has:
- Noise (cafés, traffic).
- Multiple speakers, overlap.
- Domain shift (medical jargon, code-switching).

WER on the production-audio test set is what matters — typically 2-3× the LibriSpeech-clean number. Plus per-utterance WER distribution, not just mean — a long-tail of high-WER utterances matters more than mean for user experience.

---

## Q12. "QA system — F1 score is 75, EM is 60. Diagnosis?"

**Answer.** F1 measures token overlap with the gold answer; EM is exact string match. F1 > EM by 15 means many predictions are *partially* right — different word boundaries, includes/excludes determiners, paraphrases. Two responses:
1. Tighten extraction (start/end logits) — F1 should converge with EM.
2. Or: accept EM 60 if downstream consumes the F1-style overlap; F1 is the user-facing answer in many systems.

---

## Q13. "Open-ended generation — what metrics?"

**Answer.** Hard. No single number works.
- **BERTScore** for semantic similarity to a reference (if one exists).
- **Perplexity** of the output under a separate LM (fluency proxy).
- **Self-BLEU** across multiple samples (diversity — high self-BLEU = repetitive).
- **Reward model scores** trained on human preferences.
- **Human pairwise eval** on every release.

Default position: human eval is the gold; everything else is a tracking proxy. Don't ship on metrics alone.

---

## Q14. "How does TorchMetrics handle DDP for BERTScore?"

**Answer.** State is per-rank token-embedding similarity scores. `_sync_dist` gathers via `all_gather_object`. Cost per batch is moderate; per epoch on a 50k-sentence eval set, gather is fine. The bigger cost: BERT forward passes — they're done on each rank in parallel, no DDP overhead there.

---

## Q15. "Translation quality estimation — explain COMET vs BLEU."

**Answer.** BLEU is a *reference-based* metric — needs ground-truth translations. COMET is a *learned* metric: a regression model trained on human-judgment data, takes (source, prediction, reference) or (source, prediction) and predicts a quality score. Better correlation with human judgments, especially in low-resource and stylistic settings. Modern WMT now reports both. TorchMetrics has BLEU/SacreBLEU/chrF; COMET requires the `unbabel-comet` package.

---

## Q16. "What's the most common BLEU bug?"

**Answer.** Passing references the wrong way. TorchMetrics expects `bleu(predictions, references)` where `references` is a list of *list of strings* (one inner list per prediction, supporting multiple references). Single-reference users often pass `bleu(preds, refs)` with `refs` flat → metric reads each ref as multiple references for one pred and the alignment is wrong. Always wrap: `bleu(preds, [[r] for r in refs])`.

---

[← Back to family page](./index.md)
