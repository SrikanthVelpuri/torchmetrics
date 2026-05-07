# Deep Dive: Questions, Math, Probability, and Research Behind the Amazon Scenarios

This page explains the reasoning process behind the Amazon scenario pack. The goal is not just to pick a metric after
the fact. The goal is to ask the right questions **before** solving the problem, and to make sure those questions are
anchored in decision theory, probability, queueing math, and well-known evaluation research.

Each scenario page in this pack now applies the same reasoning pattern directly in place:

- `Worked Example`
- `Questions I Asked Before Solving`
- `Why Those Questions Are Backed by Math, Probability, or Research`
- `How the Questions Changed the Final Metric and Threshold Choice`

This shared page stays intentionally more general. The scenario pages show how the same reasoning becomes
decision-specific once the workflow, cost surface, and queue constraints are known.

## 1. The questions to ask before solving any scenario

Before choosing a model, threshold, or metric, I ask these questions:

### Question 1: What decision will the model trigger?

This is the first question because the metric must match the decision.

- Will the score trigger a **hard action** such as hold, suppress, or auto-route?
- Will it create a **review queue**?
- Will it act as a **top-k assistant** for a human?
- Will the score itself be interpreted as a **probability**?

The mathematical reason is simple:

```text
best_action(x) = argmin_a sum_y P(y | x) * Cost(a, y)
```

If the system is actually a queue-ranking problem, then a threshold-free ranking metric matters. If it is a
hard-decision problem, then thresholded confusion-matrix outcomes matter. If humans consume probabilities directly,
calibration matters.

### Question 2: What is the base rate?

This is the prevalence question:

```text
prevalence = positives / total_examples
```

This question is backed by probability because **precision depends directly on prevalence**. In rare-event systems,
strong accuracy or AUROC can still produce unusable precision.

Example:

```text
daily_volume = 1,000,000
abuse_prevalence = 0.1%
positives = 1,000
negatives = 999,000

if recall = 90% and specificity = 99%:
TP = 900
FP = 9,990
precision = 900 / (900 + 9,990) = 8.3%
```

That is why an "excellent" specificity can still create a broken investigation queue.

### Question 3: What is the cost of false positives and false negatives?

This question is backed by expected-value mathematics:

```text
expected_cost(threshold)
= FP(threshold) * cost_false_positive
+ FN(threshold) * cost_false_negative
```

If the scores are calibrated probabilities and the correct-decision cost is near zero, then the decision threshold is
not arbitrary. A useful approximation is:

```text
take_positive_action if P(y=1 | x) >= cost_FP / (cost_FP + cost_FN)
```

This is why a seller-hold threshold, review threshold, and suppression threshold are business-policy choices, not just
model defaults.

### Question 4: What queue capacity can the operation sustain?

This question is backed by queueing math. A simple form is:

```text
alerts_per_day = total_volume * predicted_positive_rate
backlog_growth = arrivals_per_day - reviews_completed_per_day
```

And the queueing intuition from Little's Law is:

```text
L = lambda * W
```

where `L` is average queue size, `lambda` is arrival rate, and `W` is average time in system.

If a threshold creates a queue faster than the team can clear it, the model is not production-ready, even if offline
metrics improve.

### Question 5: When do labels become trustworthy?

This is the label-maturity question.

- Are fraud labels mature only after investigation?
- Are returns labels mature only after the return window closes?
- Are moderation labels mature only after appeal resolution?

This is backed by statistical validity. If labels are immature, then precision, recall, and calibration are biased
because the "truth" is incomplete.

### Question 6: Which slices can fail silently?

This is the subgroup-risk question.

- Which marketplace, category, seller cohort, carrier, or language slice has the most operational risk?
- Do I have enough positive support in each slice to trust the estimate?

This is backed by probability and uncertainty estimation. For a slice with only a small number of positives, an
estimate can be too noisy to trust.

Approximate standard error for a proportion:

```text
SE(p_hat) ~= sqrt(p_hat * (1 - p_hat) / n)
```

So if a slice has very few positive examples, the interval around recall or precision can be extremely wide.

### Question 7: Is the real workflow top-1, top-k, ranking, or calibrated probability?

This question is backed by task design.

- If the user sees only one answer, **top-1** matters.
- If the user sees a shortlist, **top-k** matters.
- If a reviewer processes a queue, **ranking metrics** matter.
- If business teams consume risk bands, **calibration** matters.

This is why multiclass routing and multilabel moderation often fail when they are flattened into one aggregate metric.

## 2. How I made sure the questions were backed by research, mathematics, and probability

I used four checks before trusting a question:

### Check 1: I translated the question into a formula

If I could not turn the question into one of these objects, it was probably too vague:

- a prevalence equation
- a confusion-matrix equation
- an expected-cost equation
- a queue-capacity equation
- a calibration relation
- a confidence or uncertainty estimate

### Check 2: I asked whether the question changes the deployed action

A good question changes one of these:

- the chosen metric
- the threshold
- the slice gating
- the queue design
- the rollback rule

If the answer would not change any operational decision, it was usually not the right first question.

### Check 3: I checked whether evaluation research supports the metric choice

The research-backed rules used in this pack are:

- Precision-recall analysis is more informative than ROC-style thinking in heavily imbalanced settings.
- Calibrated probabilities are necessary when scores drive decisions or risk bands.
- Cost-sensitive decision-making should use expected cost, not only accuracy.
- Queue-constrained operations need threshold simulation, not only offline ranking quality.

### Check 4: I checked whether the question survives a simple numerical example

If a question is important, a small numeric example should reveal why.

Examples:

- Can 99% specificity still break a review queue under 0.1% prevalence? Yes.
- Can a model with excellent micro metrics still fail rare severe labels? Yes.
- Can top-1 accuracy look fine while the real agent-assist value is in top-3? Yes.

## 3. Seller Risk Deep Dive

### Clear example

Suppose Amazon sees `1,000,000` seller-related events per day, and confirmed abuse prevalence on mature labels is
`0.1%`.

If the model reaches:

```text
recall = 90%
specificity = 99%
```

then:

```text
TP = 900
FP = 9,990
precision = 8.3%
```

That means the main operational question is not "Is the model good?" The real question is "Can the analyst queue
survive this precision?"

### Questions I asked before solving

- What is the mature abuse prevalence by marketplace and seller cohort?
- What precision floor does the analyst queue require?
- What is the cost of a false hold relative to missed abuse?
- Does a score of `0.90` really correspond to a very high observed abuse rate?
- Which cohorts create the most damage if recall drops?

### Why those questions are mathematically justified

- The prevalence question is justified because precision is prevalence-sensitive.
- The analyst-capacity question is justified by queueing math: `queue_in = TP + FP`.
- The false-hold question is justified by expected cost: seller friction is a real `cost_FP`.
- The score-band question is justified by calibration: if `P(y=1|score~=0.9)` is not actually high, risk bands are misleading.
- The cohort question is justified by conditional probability: `P(abuse | new_seller)` and `P(abuse | established_seller)` can differ sharply.

## 4. Returns Abuse Deep Dive

### Clear example

Suppose the system processes `100,000` returns in a period, with mature abuse prevalence of `1.5%`.

If a threshold gives:

```text
recall = 80%
specificity = 95%
```

then:

```text
positives = 1,500
negatives = 98,500
TP = 1,200
FP = 4,925
precision = 19.6%
```

That may still be too noisy if manual review is expensive or if false positives delay good-customer refunds.

### Questions I asked before solving

- When do abuse labels become mature for each category?
- Do apparel, electronics, and consumables need different thresholds?
- How much customer-friction cost does one false positive create?
- How many reviewed returns per day can the team absorb during peak season?
- Is the score being used as a direct risk band for policy decisions?

### Why those questions are mathematically justified

- Label maturity matters because biased labels distort recall and calibration.
- Category-specific thresholding is justified because `cost_FP` and `cost_FN` are example-dependent.
- Customer-friction cost belongs directly in the expected-cost equation.
- Queue-size questions follow from `review_load = TP + FP`.
- Risk-band usage makes calibration mathematically necessary.

## 5. Catalog Quality Deep Dive

### Clear example

Assume a multilabel system with two tags:

- common attribute tag prevalence = `40%`
- rare safety tag prevalence = `0.2%`

If the model is excellent on the common tag and weak on the rare tag, micro metrics can still look strong because the
common label dominates the count.

That is why one of the first questions is: "Which labels are rare but operationally critical?"

### Questions I asked before solving

- Which labels are rare but high severity?
- Are humans consuming final tags or ranked candidate tags?
- What is the average number of true labels per listing?
- Are thresholds shared across labels, or should they vary by severity?
- Has the taxonomy changed recently?

### Why those questions are mathematically justified

- Rare-label importance is a weighting question, which is why macro averaging matters.
- Ranked-review questions justify ranking metrics such as ranking average precision and coverage error.
- Label-cardinality questions matter because multilabel noise changes with the expected number of positive tags.
- Shared-threshold questions are cost-sensitive decision questions.
- Taxonomy drift changes the probability space itself, so historical comparisons may stop being valid.

## 6. Support Routing Deep Dive

### Clear example

Suppose top-1 accuracy is `82%`, but top-3 accuracy is `97%`.

If the workflow is agent assist, the second number may matter far more than the first one.

Now suppose most mistakes are harmless between two low-cost queues, but a small fraction route fraud cases into a
general-support queue. Then the confusion matrix matters more than the headline accuracy.

### Questions I asked before solving

- Is the workflow full automation or agent assist?
- Which class confusions are cheap, and which are operationally expensive?
- Do we need an abstain band for low-confidence predictions?
- Are queue definitions stable across time and locales?
- Does the evaluation include the languages and marketplaces that matter live?

### Why those questions are mathematically justified

- Full-automation versus assistive design determines whether top-1 or top-k is the right success event.
- Expensive confusions imply a cost matrix, not a uniform 0/1 loss.
- Abstention depends on calibrated confidence.
- Taxonomy-stability questions protect label validity.
- Locale coverage is a conditional-distribution question: `P(intent | language)` can change substantially.

## 7. Review Moderation Deep Dive

### Clear example

Suppose:

- spam prevalence is high
- severe self-harm or dangerous-product labels are very rare

If the model improves micro metrics by getting better at spam while staying weak on rare severe labels, production has
not improved in the way policy teams care about.

### Questions I asked before solving

- Which labels are severe enough to need their own precision floor?
- Are reviewers consuming ranked items or final hard actions?
- How much reviewer capacity exists per day?
- Which locales or languages have distinct policy behavior?
- How quickly do appeal outcomes change the effective label distribution?

### Why those questions are mathematically justified

- Severe labels justify per-label thresholds because their `cost_FN` is far larger.
- Ranked-review questions justify ranking metrics and coverage error.
- Reviewer-capacity questions follow queueing math.
- Locale questions are slice-probability questions.
- Appeal-driven changes affect both label quality and calibration.

## 8. Delivery Exception Deep Dive

### Clear example

Suppose `96%` of orders are on time.

A model with `96.5%` top-1 accuracy can still be poor if it misses the rare exception classes that actually drive
expedited shipping, customer contacts, and promise breaches.

### Questions I asked before solving

- What is the prevalence of each exception class by station and carrier?
- Is the operation using top-1 action or top-2 diagnosis support?
- How costly is a false intervention compared with a missed true exception?
- Are station labels consistent enough to trust as ground truth?
- Will one threshold overload certain regions during storms or peak periods?

### Why those questions are mathematically justified

- Rare-class prevalence justifies macro metrics.
- Top-2 support questions justify top-k metrics.
- Intervention tradeoffs are expected-cost questions.
- Label-consistency questions justify agreement-oriented diagnostics such as Cohen kappa.
- Region-overload questions are queue-capacity questions under changing class priors.

## 9. A short decision checklist I would use before solving any new Amazon scenario

Use this checklist before you write a line of training code:

1. What exact action will the model trigger?
2. What is the mature base rate on recent data?
3. What are the costs of FP and FN in business terms?
4. How many actions per day can the operation absorb?
5. Are probabilities used as probabilities, or only for ranking?
6. Which slices are too important to hide inside the global average?
7. How many positive examples exist in those slices?
8. Has the label definition or taxonomy changed?
9. Does the live workflow care about top-1, top-k, ranking, or calibrated risk bands?
10. If the metric improves, which business number should also improve?

If you cannot answer question 10, the modeling problem is still not framed correctly.

## 10. Research Foundations

These references support the main reasoning patterns used in this pack:

- Charles Elkan, *The Foundations of Cost-Sensitive Learning*:
  [https://cseweb.ucsd.edu/~elkan/rescale.pdf](https://cseweb.ucsd.edu/~elkan/rescale.pdf)
- Jesse Davis and Mark Goadrich, *The Relationship Between Precision-Recall and ROC Curves*:
  [http://pages.cs.wisc.edu/~shavlik/abstracts/davis.icml06.abstract.html](http://pages.cs.wisc.edu/~shavlik/abstracts/davis.icml06.abstract.html)
- Takaya Saito and Marc Rehmsmeier, *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*:
  [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)
- Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger, *On Calibration of Modern Neural Networks*:
  [https://proceedings.mlr.press/v70/guo17a.html](https://proceedings.mlr.press/v70/guo17a.html)
- John D. C. Little, *A Proof for the Queuing Formula: L = lambda W*:
  [https://doi.org/10.1287/opre.9.3.383](https://doi.org/10.1287/opre.9.3.383)

These are not the whole story, but they are enough to justify why the questions in this pack are not arbitrary. They
come from a combination of cost-sensitive decision theory, imbalanced-learning evaluation, calibration research, and
queueing mathematics.
