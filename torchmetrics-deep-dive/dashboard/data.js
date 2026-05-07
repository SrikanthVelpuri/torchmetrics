/* ============================================================
 *  TorchMetrics Deep Dive — Dashboard data
 *  Each topic has:
 *    - id, category, title
 *    - summary: array of HTML-ish blocks (h2, p, ul, code, pre, table, callout)
 *    - questions: array of { q, a, followups: [{ q, a, followups: [...] }] }
 * ============================================================ */
window.TM_DATA = {
  categories: [
    "Foundations",
    "Domain Metrics",
    "Distributed & Lightning",
    "Custom & Production",
    "Interview Prep",
    "Business Mapping",
    "Reference",
  ],
  topics: [
    /* ============================================================ FOUNDATIONS */
    {
      id: "getting-started",
      category: "Foundations",
      title: "Getting Started",
      summary: [
        { h2: "What TorchMetrics is" },
        { p: "TorchMetrics is a PyTorch-native library of 100+ pre-built metrics plus a small base class (<code>Metric</code>) for building custom ones. It works on CPU, GPU, multi-GPU DDP, and integrates tightly with PyTorch Lightning." },
        { h2: "Two APIs" },
        { p: "<b>Functional</b>: <code>torchmetrics.functional.accuracy(preds, target)</code> — pure function, one-shot." },
        { p: "<b>Modular</b>: <code>torchmetrics.Accuracy()</code> — accumulates state across batches, syncs across DDP, reset between epochs." },
        { h2: "Minimal training loop" },
        { pre: "import torch\nfrom torchmetrics import Accuracy, MetricCollection, F1Score\n\nval_metrics = MetricCollection({\n    'acc': Accuracy(task='multiclass', num_classes=10),\n    'f1':  F1Score(task='multiclass', num_classes=10, average='macro'),\n}).to(device)\n\nfor epoch in range(5):\n    model.eval()\n    with torch.no_grad():\n        for x, y in val_loader:\n            val_metrics.update(model(x.to(device)), y.to(device))\n    print(epoch, val_metrics.compute())\n    val_metrics.reset()" },
        { h2: "Common pitfalls" },
        { ul: [
          "Forgot <code>reset()</code> — epoch 2 includes epoch 1 data.",
          "Forgot <code>.to(device)</code> — device-mismatch error.",
          "Wrong <code>task=</code> argument — binary / multiclass / multilabel are different metrics.",
          "Mixing functional + modular — averaging functional-output across batches is not the same as <code>update + compute</code>.",
        ]},
      ],
      questions: [
        {
          q: "When do you choose the functional API over the modular one?",
          a: "Functional for one-shot computation on a fixed (preds, target) pair, inside <code>torch.no_grad()</code>, when you don't need to accumulate or sync. Modular for streaming across batches or running in DDP / Lightning.",
          followups: [
            {
              q: "Why is averaging functional outputs across batches wrong for many metrics?",
              a: "Most useful metrics (F1, AUROC, mAP) are non-decomposable — averaging per-batch values doesn't equal the global value. Modular metrics aggregate sufficient statistics, then compute once.",
              followups: [
                {
                  q: "Give a concrete example where averaging breaks accuracy.",
                  a: "Batch 1: 90/100 correct (acc=0.90). Batch 2: 1/5 correct (acc=0.20). Mean of batch accuracies = 0.55. Correct global = 91/105 = 0.867.",
                  followups: [
                    {
                      q: "What's the fix?",
                      a: "Aggregate sufficient statistics: keep <code>correct</code> and <code>total</code> across batches; divide once at the end. That's exactly what modular <code>Accuracy()</code> does."
                    }
                  ]
                }
              ]
            }
          ]
        },
        {
          q: "Calling a metric like a function — <code>metric(preds, target)</code> — what does it return?",
          a: "It calls <code>forward()</code>, which (a) updates the running state with this batch and (b) returns the metric value computed on this batch only.",
          followups: [
            {
              q: "Is the per-batch value 'correct' for non-decomposable metrics?",
              a: "Yes — it's the metric computed as if this batch were the entire eval set. Useful for logging step-level training curves; not the same as the running global value.",
              followups: [
                {
                  q: "How does TorchMetrics produce both efficiently?",
                  a: "Two paths: <code>_forward_full_state_update</code> (safe, two updates per call) or <code>_forward_reduce_state_update</code> (fast, one update plus state-merge). The base class picks based on <code>full_state_update</code>."
                }
              ]
            }
          ]
        },
        {
          q: "Why does the new classification API require <code>task=</code>?",
          a: "Old API accepted <code>num_classes</code> only and silently picked behavior. New API is explicit: <code>BinaryF1Score</code>, <code>MulticlassF1Score</code>, <code>MultilabelF1Score</code> — different math. The <code>F1Score(task='binary')</code> form is a thin wrapper for backward compatibility.",
          followups: [
            {
              q: "Why was the old behavior dangerous?",
              a: "A multilabel target accidentally passed to a multiclass metric returned a number — silently wrong. Explicit task makes the error loud."
            }
          ]
        },
      ]
    },

    {
      id: "core-concepts",
      category: "Foundations",
      title: "Core Concepts",
      summary: [
        { h2: "The four primitives" },
        { ul: [
          "<code>add_state(name, default, dist_reduce_fx)</code> — declare state.",
          "<code>update(*args)</code> — mutate state from a batch (no return).",
          "<code>compute()</code> — pure function from state to value.",
          "<code>reset()</code> — restore state to declared defaults.",
        ]},
        { h2: "Lifecycle" },
        { p: "<code>__init__</code> registers state (deepcopy'd defaults). <code>update</code> is wrapped to bump <code>_update_count</code> and clear cache. <code>compute</code> is wrapped to sync DDP and cache. <code>reset</code> restores defaults so the same metric can be reused next epoch." },
        { h2: "State types" },
        { table: [
          ["Default", "Behavior", "DDP reduction"],
          ["<code>tensor(0.0)</code>", "Mutated in place each update.", "sum, mean, min, max, cat"],
          ["<code>[]</code>", "Append tensors during update.", "Almost always cat"],
        ]},
        { callout: "Tensor states are O(1) memory and DDP-friendly. List states grow with eval-set size and need <code>compute_on_cpu=True</code> for large evals." },
        { h2: "Two forward modes" },
        { ul: [
          "<code>full_state_update=True</code>: safe; two <code>update</code> calls per <code>forward</code>.",
          "<code>full_state_update=False</code>: fast; one update + state merge via <code>dist_reduce_fx</code>.",
        ]},
        { h2: "Performance flags" },
        { table: [
          ["Flag", "Default", "Purpose"],
          ["compute_on_cpu", "False", "Move list states to CPU after each update."],
          ["dist_sync_on_step", "False", "Sync state every <code>forward</code>; almost always wrong."],
          ["sync_on_compute", "True", "Sync state inside <code>compute</code>."],
          ["compute_with_cache", "True", "Cache <code>compute</code> result until next <code>update</code>."],
        ]},
      ],
      questions: [
        {
          q: "Why does TorchMetrics split state from compute?",
          a: "Most ML metrics are non-decomposable across batches and devices. Aggregating raw <i>state</i> (sufficient statistics) is summable; aggregating <i>output values</i> is not. <code>update</code> mutates state; <code>compute</code> is pure math from state to value.",
          followups: [
            {
              q: "Give an example where state-level aggregation matters under DDP.",
              a: "F1 across 8 GPUs: each rank computes its TP/FP/FN. Mean of 8 per-rank F1 values is wrong. Sum of TP/FP/FN across ranks then computing F1 is right.",
              followups: [
                {
                  q: "How does <code>add_state</code> support this?",
                  a: "It registers a per-state reduction (<code>dist_reduce_fx='sum'</code> for counts). On <code>compute</code>, the base class <code>all_gather</code>s state across ranks and applies that reduction.",
                  followups: [
                    {
                      q: "What if I forgot the reduction?",
                      a: "<code>_reductions[name] = None</code>. After <code>all_gather</code>, the state has shape <code>(world_size, ...)</code> and <code>compute</code> treats the rank dim as data — silent miscalculation. Always specify the reduction."
                    }
                  ]
                }
              ]
            }
          ]
        },
        {
          q: "When can you safely set <code>full_state_update=False</code>?",
          a: "When every state's reduction is one of the standard ones (sum / mean / cat / min / max). The fast forward path then merges the saved global state with the batch state via that reduction.",
          followups: [
            {
              q: "What's the speedup?",
              a: "About 2× on <code>forward()</code> calls — one <code>update</code> per call instead of two. For metrics in a hot training-step log, this matters."
            }
          ]
        },
        {
          q: "Why do list states blow up memory?",
          a: "They store every prediction and target until <code>compute</code>. <code>all_gather</code> ships everything to every rank. A 10M-sample AUROC eval with 8 GPUs = 80M-row tensors momentarily on each rank.",
          followups: [
            {
              q: "How do you fix it without changing the metric?",
              a: "<code>compute_on_cpu=True</code> moves the list states to host RAM after every update. Trades PCIe traffic for GPU RAM.",
              followups: [
                {
                  q: "When does that not fix it?",
                  a: "When even host RAM is too small. Then switch to a binned variant (tensor-state approximation) or shard eval and merge via <code>merge_state</code> offline."
                }
              ]
            }
          ]
        },
      ]
    },

    {
      id: "metric-internals",
      category: "Foundations",
      title: "Metric Class Internals",
      summary: [
        { h2: "Metric subclasses Module + ABC" },
        { p: "Inherits from <code>torch.nn.Module</code> so <code>.to()</code>, <code>state_dict</code>, hooks all work. Inherits from <code>ABC</code> so <code>update</code> and <code>compute</code> must be implemented." },
        { h2: "What __init__ does" },
        { ul: [
          "Parses + validates kwargs (<code>compute_on_cpu</code>, <code>sync_on_compute</code>, etc.).",
          "Wraps <code>self.update</code> and <code>self.compute</code> so the base class can intercept.",
          "Initializes three parallel dicts: <code>_defaults</code>, <code>_persistent</code>, <code>_reductions</code>.",
          "Calls <code>torch._C._log_api_usage_once</code> for PyTorch's anonymous telemetry.",
        ]},
        { h2: "_wrap_update" },
        { pre: "def wrapped(*args, **kwargs):\n    self._computed = None\n    self._update_count += 1\n    with torch.set_grad_enabled(self._enable_grad):\n        try:\n            update(*args, **kwargs)\n        except RuntimeError as err:\n            if 'Expected all tensors to be on' in str(err):\n                # rewrite with friendlier hint\n                raise RuntimeError('... try .to(device) ...') from err\n            raise" },
        { h2: "_sync_dist (the DDP magic)" },
        { ul: [
          "Pre-concatenates list states locally (one all_gather per state).",
          "Calls <code>gather_all_tensors</code> — handles ragged shapes via padding.",
          "Applies registered reduction (<code>dim_zero_sum</code>, <code>dim_zero_cat</code>, …).",
          "Adds empty placeholder for ranks with no data.",
        ]},
        { h2: "compute is wrapped to" },
        { ul: [
          "Warn if <code>_update_count == 0</code>.",
          "Return cached value if available.",
          "Sync inside a context that <code>unsync</code>s on exit.",
          "Cache result until next <code>update</code>.",
        ]},
      ],
      questions: [
        {
          q: "Walk me through a <code>compute()</code> call on rank 3 of an 8-GPU job.",
          a: "(1) Wrapper checks <code>_update_count > 0</code>; warns if not. (2) Returns cached if available. (3) Calls <code>sync()</code>: copies state to <code>_cache</code>, calls <code>_sync_dist</code>. (4) <code>_sync_dist</code> pre-concats list states locally, <code>all_gather</code>s, applies reduction. (5) Runs user <code>compute</code> on merged state. (6) <code>unsync()</code> restores local state from cache. (7) Returns and caches result.",
          followups: [
            {
              q: "Why restore local state on unsync?",
              a: "So subsequent <code>update()</code>s keep accumulating local data correctly. Without unsync, the next update would add to already-merged state on every rank — double-counting on next compute.",
              followups: [
                {
                  q: "Could you ever want to skip unsync?",
                  a: "Yes — pass <code>should_unsync=False</code> to the manual <code>sync()</code> API when you want every rank to have the global state for further derived computation.",
                  followups: [
                    {
                      q: "Concrete situation?",
                      a: "Computing class-balanced accuracy from a globally-merged confusion matrix as a derived step. You sync, compute the derived metric, then <code>reset()</code> for the next epoch."
                    }
                  ]
                }
              ]
            }
          ]
        },
        {
          q: "Why is the <code>update</code> wrapper catching device-mismatch errors and rewriting them?",
          a: "Because it's the single most common user error — metric on CPU, inputs on GPU. The original PyTorch error is cryptic; the wrapper rewrites it to point at <code>metric.to(device)</code>, saving a lot of issue-tracker time.",
          followups: [
            {
              q: "Why not just auto-move the metric to the input's device?",
              a: "Silent magic. If the user's intent was to keep the metric on CPU (e.g. for a CPU-side aggregation), auto-moving would surprise them. Explicit error + hint is safer."
            }
          ]
        },
        {
          q: "What does <code>add_state</code> actually store?",
          a: "Three things, in three parallel dicts: <code>_defaults[name]</code> = deepcopy of default (for <code>reset</code>); <code>_persistent[name]</code> = whether it goes in <code>state_dict</code>; <code>_reductions[name]</code> = the DDP reduction callable. The current value lives as <code>self.&lt;name&gt;</code> via <code>setattr</code>.",
          followups: [
            {
              q: "Why deepcopy the default?",
              a: "So <code>reset()</code> can restore without sharing state across instances. Without deepcopy, two metrics constructed with the same default tensor would share that buffer.",
              followups: [
                {
                  q: "What about list defaults?",
                  a: "List defaults must be empty <code>[]</code>. The deepcopy is a fresh empty list. List elements (tensors) are appended in <code>update</code>; on <code>reset</code>, the attribute is replaced with a new empty list — old elements become unreferenced."
                }
              ]
            }
          ]
        },
        {
          q: "Why is there a 'caching' layer on <code>compute</code>?",
          a: "So calling <code>compute()</code> twice in a row is O(1). It returns the cached value until <code>update</code> invalidates it. Useful when multiple consumers want the same number (logger + early-stopping + dashboard) without recomputing.",
          followups: [
            {
              q: "When would you disable caching?",
              a: "If your <code>compute</code> depends on something other than the declared state (e.g. reads global mutable state, randomness). Pass <code>compute_with_cache=False</code>.",
            }
          ]
        },
      ]
    },

    /* ============================================================ DOMAIN METRICS */
    {
      id: "classification",
      category: "Domain Metrics",
      title: "Classification Metrics",
      summary: [
        { h2: "Task taxonomy" },
        { table: [
          ["Task", "Each sample", "preds shape", "target shape"],
          ["binary", "1 of 2", "(N,) float / int", "(N,) {0,1}"],
          ["multiclass", "1 of K", "(N, K) logits or (N,) ints", "(N,) [0,K)"],
          ["multilabel", "subset of K", "(N, K) logits/probs", "(N, K) {0,1}"],
        ]},
        { h2: "StatScores foundation" },
        { p: "Most metrics derive from TP/FP/TN/FN counts. Accuracy, Precision, Recall, F1, MCC, balanced accuracy — all from those four numbers. <code>MetricCollection</code> shares this state via compute groups." },
        { h2: "Average modes" },
        { ul: [
          "<code>micro</code>: pool counts globally, then compute. Tracks dominant class.",
          "<code>macro</code>: per-class metric, then unweighted mean. Treats every class equally.",
          "<code>weighted</code>: per-class weighted by support.",
          "<code>None</code>: per-class array.",
        ]},
        { h2: "Curve metrics (list-state)" },
        { p: "AUROC, Average Precision, ROC, Precision-Recall Curve, Calibration Error, LogAUC. Need full prediction distribution, not just TP/FP." },
      ],
      questions: [
        {
          q: "AUROC vs Average Precision — when do you prefer which?",
          a: "AUROC integrates over FPR — dominated by negatives. AP integrates over recall on positives. <b>Imbalanced ⇒ AP. Balanced ⇒ either, AP safer.</b>",
          followups: [
            {
              q: "Will TorchMetrics' AUROC match sklearn?",
              a: "Yes, to ~1e-6, enforced by parity tests. Tie-breaking matches sklearn's step interpolation for AP.",
              followups: [
                {
                  q: "What if you can't fit predictions in GPU memory?",
                  a: "<code>compute_on_cpu=True</code> for list-state, or switch to a binning approximation: bucket predictions into K bins, track <code>(positives_per_bin, count_per_bin)</code>. Tensor-state, summable, slightly approximate.",
                  followups: [
                    {
                      q: "How approximate?",
                      a: "K=10000: ~1e-4 error. K=1000: ~1e-3. K=100: meaningful error. Use for monitoring; never for final reporting at K<5000."
                    }
                  ]
                }
              ]
            }
          ]
        },
        {
          q: "Why do macro and micro F1 differ on the same data?",
          a: "Micro pools TP/FP/FN across classes — tracks the dominant class. Macro per-class then unweighted-averages — every class counts equally. On imbalanced data they're very different numbers.",
          followups: [
            {
              q: "Which goes in the paper?",
              a: "Both, with a footnote on imbalance. Hiding either gets caught in review.",
              followups: [
                {
                  q: "Macro silently weights a 1-sample class equal to a 1M-sample class. Bug or feature?",
                  a: "Feature when long-tail performance matters (rare-disease detection). Bug when classes are spurious. Use 'weighted' when class size is the right importance."
                }
              ]
            }
          ]
        },
        {
          q: "Why isn't softmax 'good enough' for calibration?",
          a: "Cross-entropy training optimizes likelihood, not calibration. Modern deep nets are systematically over-confident. Calibration error (ECE) measures the gap between predicted confidence and observed accuracy.",
          followups: [
            {
              q: "What do you do when ECE is bad?",
              a: "Post-hoc temperature scaling: divide logits by a learned <code>T</code> on val set. Doesn't change argmax, just sharpens / softens probabilities.",
              followups: [
                {
                  q: "When isn't temperature enough?",
                  a: "When bias is class-dependent. Then matrix scaling. TorchMetrics ships the <i>measurement</i>, not the calibrator.",
                  followups: [
                    {
                      q: "Right number of bins for ECE?",
                      a: "10–20 standard. Adaptive (equal-mass) bins are more stable than equal-width on skewed predictions."
                    }
                  ]
                }
              ]
            }
          ]
        },
        {
          q: "Multilabel with 10000 labels — what changes?",
          a: "Per-label state stays cheap (10000×4 = 40KB tensor). exact_match becomes useless. Many labels will have zero support per batch — handle empty slices. Headline metric: macro-F1 over support-positive subset, plus per-label F1 for top-100 frequent labels.",
          followups: [
            {
              q: "Aggregate macro-F1 over all 10000 — why not?",
              a: "Dominated by 9000 zero-support labels. Macro number ~ 0.05 even for a great model. Uninformative."
            }
          ]
        },
      ]
    },

    {
      id: "regression",
      category: "Domain Metrics",
      title: "Regression Metrics",
      summary: [
        { h2: "Standard error metrics" },
        { table: [
          ["Metric", "Formula", "Use"],
          ["MAE", "mean(|y-ŷ|)", "Outlier-resistant, target units"],
          ["MSE", "mean((y-ŷ)²)", "Penalizes large errors; differentiable"],
          ["RMSE", "√MSE", "Same units; report alongside MAE"],
          ["MAPE", "mean(|y-ŷ|/|y|)", "Forecasting; explodes at y≈0"],
          ["wMAPE", "Σ|err|/Σ|y|", "Robust on low-volume series"],
          ["SMAPE", "mean(2|y-ŷ|/(|y|+|ŷ|))", "Symmetric percentage error"],
        ]},
        { h2: "Goodness of fit" },
        { p: "<code>R2Score</code>: 1 − SS_res/SS_tot. Can be negative (worse than mean-predictor). Don't clamp to [0,1] — negative is an informative signal." },
        { h2: "Correlations" },
        { ul: [
          "<code>PearsonCorrCoef</code>: linear, streaming Welford-style algorithm.",
          "<code>SpearmanCorrCoef</code>: monotonic via ranks; list-state.",
          "<code>KendallRankCorrCoef</code>: rank concordance; list-state, O(n²).",
          "<code>ConcordanceCorrCoef</code>: Lin's CCC, agreement not just correlation.",
        ]},
        { h2: "Distributional / probabilistic" },
        { p: "<code>KLDivergence</code>, <code>JensenShannonDivergence</code>, <code>ContinuousRankedProbabilityScore</code>, <code>CriticalSuccessIndex</code>." },
      ],
      questions: [
        {
          q: "MAE vs MSE vs RMSE — when do you pick which?",
          a: "MAE: robust, target units, non-differentiable at zero. MSE: quadratic penalty, differentiable — usually the training loss. RMSE: same units as target. <b>Train MSE, report MAE + RMSE.</b>",
          followups: [
            {
              q: "What if your errors are heavy-tailed?",
              a: "MSE dominated by tails. Options: (a) clip targets, (b) train Huber / log-cosh, (c) train in log-space and report metrics on transformed-back values.",
              followups: [
                {
                  q: "Doesn't training in log-space distort the metric?",
                  a: "Yes — <code>MSE(log y, log ŷ)</code> is relative-error MSE. TorchMetrics has <code>MeanSquaredLogError</code>. Always report both: training-aligned metric + canonical original-space metric."
                }
              ]
            }
          ]
        },
        {
          q: "Why is MAPE dangerous in production?",
          a: "Divides by y. Near-zero rows blow up the mean. Demand series with holiday zeros, low-volume SKUs, or off-peak hours are minefields.",
          followups: [
            {
              q: "What replaces it?",
              a: "wMAPE (Σ|err|/Σ|y|) for global. SMAPE for symmetric percentage. Report both — different teams expect different defaults.",
              followups: [
                {
                  q: "SMAPE has its own pathologies — what?",
                  a: "When both y and ŷ are near zero, the denominator is also near zero. Bounded [0, 200%] but flickers wildly on near-zero values. Below a threshold, suppress percentage and report absolute error.",
                  followups: [
                    {
                      q: "If business stakeholders insist on MAPE?",
                      a: "Compute it with an explicit floor (e.g. <code>max(|y|, 1)</code>) and document the floor in the metric name (<code>MAPE_floor1</code>). Hidden flooring is malpractice."
                    }
                  ]
                }
              ]
            }
          ]
        },
        {
          q: "R² can be negative. Why doesn't TorchMetrics clamp?",
          a: "Because clamping hides bugs. Negative R² is informative — the model predicts worse than the constant-mean baseline. Coercion to 0 would silence the alarm.",
          followups: [
            {
              q: "Negative R² in production — first action?",
              a: "Page on-call. Almost always an infra issue — pipeline broken, label leakage flipped, wrong model deployed."
            }
          ]
        },
      ]
    },

    {
      id: "retrieval",
      category: "Domain Metrics",
      title: "Retrieval Metrics",
      summary: [
        { h2: "The contract" },
        { p: "Every retrieval metric expects three tensors per <code>update</code>: preds (scores), target (relevance), indexes (query id). Items are grouped by query, sorted by score, evaluated per-query, then averaged." },
        { pre: "metric.update(preds, target, indexes)\n# preds:   (N,) float scores\n# target:  (N,) int relevance (binary or graded)\n# indexes: (N,) int query id" },
        { h2: "Available metrics" },
        { ul: [
          "<code>RetrievalMRR</code> — 1 / rank_of_first_relevant.",
          "<code>RetrievalMAP</code> — area under per-query PR curve.",
          "<code>RetrievalNDCG</code> — graded relevance, position-discounted.",
          "<code>RetrievalPrecision</code> / <code>RetrievalRecall</code> @ k.",
          "<code>RetrievalHitRate</code> — any relevant in top-k?",
          "<code>RetrievalRPrecision</code>, <code>RetrievalAUROC</code>, <code>RetrievalFallOut</code>.",
        ]},
        { h2: "Empty-target action" },
        { p: "<code>empty_target_action='skip'|'neg'|'pos'|'error'</code>. Default behavior changes the reported number meaningfully — pick deliberately." },
      ],
      questions: [
        {
          q: "Why does retrieval need <code>indexes</code>?",
          a: "Retrieval evaluates per-query lists, not per-sample independently. Without indexes, the metric treats the whole batch as one giant query — almost never what you want.",
          followups: [
            {
              q: "Walk through what compute() does with indexes.",
              a: "(1) Group rows by index. (2) Sort each group by predicted score descending. (3) Compute per-query metric. (4) Aggregate (mean by default). All steps require list state."
            }
          ]
        },
        {
          q: "NDCG vs MAP — when do you pick which?",
          a: "MAP assumes binary relevance, integrates precision-at-each-relevant-rank. NDCG handles graded relevance via gain formula and discount. <b>Graded ⇒ NDCG. Binary ⇒ either, NDCG more flexible.</b>",
          followups: [
            {
              q: "Why discount = 1/log2(rank+1) specifically?",
              a: "Empirical: matches user click curves across many domains. You can change it for your business; the math still works for any monotonically decreasing discount.",
              followups: [
                {
                  q: "Could you fit the discount to your platform's click data?",
                  a: "Yes — fit position-click curve from logs, plug into a custom NDCG. Caveat: literature comparability is lost.",
                  followups: [
                    {
                      q: "What about position bias in click data?",
                      a: "Click data has positional bias (top results clicked because they're top). Correct via inverse-propensity-scoring before fitting, or use interleaving experiments."
                    }
                  ]
                }
              ]
            }
          ]
        },
        {
          q: "Recall@1000 vs Recall@10 — what's the architectural significance?",
          a: "Different stages. Recall@1000 evaluates first-stage candidate generation (don't lose relevant items). Recall@10 evaluates the reranker (top-k ordering quality).",
          followups: [
            {
              q: "What recall@1000 do you want?",
              a: "≥ 95–98 % for a 1k-candidate reranker. Below 90 %, the reranker is bottlenecked by what got recalled — it can't fix items it never sees.",
              followups: [
                {
                  q: "How do you pick K for the candidate set?",
                  a: "Sweep K = {100, 500, 1000, 5000}; plot recall@K and reranker latency. Smallest K where recall plateaus and latency fits budget. Often a knee at K≈1000."
                }
              ]
            }
          ]
        },
        {
          q: "<code>empty_target_action</code> — what's the right default?",
          a: "<code>'skip'</code> is most defensible (don't penalize for queries with no ground-truth-positive). <code>'neg'</code> is stricter. <code>'pos'</code> rarely correct.",
          followups: [
            {
              q: "When is 'skip' wrong?",
              a: "When 'no relevant items' is itself a valid signal. Spam detection: a query with no spam should result in zero pulled. Reporting that as 'skipped' hides the system's correct behavior."
            }
          ]
        },
      ]
    },

    {
      id: "tai-metrics",
      category: "Domain Metrics",
      title: "Text, Audio, Image",
      summary: [
        { h2: "Text" },
        { ul: [
          "BLEU (precision, n-gram, brevity penalty) — MT.",
          "ROUGE (recall) — summarization.",
          "chrF, METEOR, TER — alternatives.",
          "WER, CER, MER — ASR.",
          "Perplexity — LM eval (token log-probs).",
          "BERTScore, InfoLM — embedding-based, GPU-heavy.",
        ]},
        { h2: "Audio" },
        { ul: [
          "SI-SDR, SDR, SNR — source separation, differentiable.",
          "PESQ, STOI — perceptual / intelligibility, non-differentiable.",
          "DNSMOS, NISQA — learned MOS predictors.",
        ]},
        { h2: "Image" },
        { ul: [
          "PSNR, SSIM, MS-SSIM — pixel/structural reconstruction.",
          "FID, KID, IS — feature-distribution distances (generative eval).",
          "LPIPS — learned perceptual similarity.",
          "CLIPScore — text-image alignment.",
        ]},
      ],
      questions: [
        {
          q: "BLEU vs ROUGE — when?",
          a: "BLEU precision-oriented (n-gram, brevity penalty) — MT. ROUGE recall-oriented — summarization. Different jobs, different metrics.",
          followups: [
            {
              q: "Why does sentence-BLEU give weird numbers?",
              a: "Geometric mean of n-gram precisions is fragile on short text. Smoothing helps but sentence-BLEU is widely considered unreliable. Always prefer corpus-level reporting.",
              followups: [
                {
                  q: "What for sentence-level MT eval?",
                  a: "chrF (CHRFScore) more stable than sentence-BLEU. Or BERTScore for semantic similarity. Modern WMT uses learned metrics like COMET (not in TorchMetrics)."
                }
              ]
            }
          ]
        },
        {
          q: "FID at 1k samples vs 50k samples — why differ?",
          a: "FID is biased downward at small N — covariance estimates have high variance. Bias decreases as 1/N. Two papers comparing at different N are not comparable.",
          followups: [
            {
              q: "How to mitigate small-N bias?",
              a: "KID (Kernel Inception Distance). MMD estimator unbiased at small N. Report both; KID alone isn't the literature standard, but it's the honest answer at low sample count.",
              followups: [
                {
                  q: "What's the right report?",
                  a: "Both, with N. <i>'FID@10k = X, KID@10k = Y'</i> beats either alone."
                }
              ]
            }
          ]
        },
        {
          q: "Why is BERTScore expensive?",
          a: "Each call runs a transformer (default <code>roberta-large</code>) over reference and hypothesis pairs. ~1 GFLOP per token-pair. Bottleneck for any text eval at scale.",
          followups: [
            {
              q: "Cheaper without losing fidelity?",
              a: "Compute reference BERTScore-large once on a held-out set. Compute it with a smaller model on the same set. Train a calibration regression from small to large. Use calibrated small-model in production.",
              followups: [
                {
                  q: "Doesn't that lose interpretability?",
                  a: "Yes — your numbers no longer compare to literature BERTScore-large. Document this. Use cheap-version only for relative model-vs-model comparisons, not absolute reporting."
                }
              ]
            }
          ]
        },
        {
          q: "PESQ — why not differentiable?",
          a: "Wraps an ITU-T C reference implementation: psychoacoustic filtering, time alignment, disturbance modeling — none of which are designed to be differentiable.",
          followups: [
            {
              q: "What do you use as a training loss for speech enhancement?",
              a: "SI-SDR (differentiable, scale-invariant). NegSTOI variants exist as differentiable surrogates.",
              followups: [
                {
                  q: "Train SI-SDR but report PESQ — why?",
                  a: "Different audiences. Researchers train on what optimizes well; product reports what reviewers know. SI-SDR ↔ PESQ correlation is high but not 1.0; you trust that correlation."
                }
              ]
            }
          ]
        },
      ]
    },

    /* ============================================================ DISTRIBUTED & LIGHTNING */
    {
      id: "ddp",
      category: "Distributed & Lightning",
      title: "Distributed Training (DDP)",
      summary: [
        { h2: "The problem" },
        { p: "On 8 GPUs, each rank sees 1/8 of the eval set. Per-rank metrics can't be averaged for non-decomposable metrics (F1, AUROC). Sum sufficient statistics first, then compute." },
        { h2: "How TorchMetrics fixes it" },
        { ul: [
          "Every state has a <code>dist_reduce_fx</code>.",
          "<code>compute()</code> wrapper calls <code>_sync_dist</code>, which <code>all_gather</code>s state.",
          "Standard reductions (sum, mean, cat, min, max) handle 90 % of cases.",
          "<code>gather_all_tensors</code> pads ragged shapes for <code>all_gather</code>.",
          "Empty rank? Send <code>tensor([], device=..., dtype=...)</code>.",
          "Local state restored from cache after compute (non-destructive).",
        ]},
        { h2: "Sync controls" },
        { table: [
          ["Setting", "Effect"],
          ["sync_on_compute=True (default)", "all_gather on every compute()"],
          ["dist_sync_on_step=True", "all_gather on every forward() — almost always wrong"],
          ["sync_on_compute=False", "compute() returns local rank only"],
        ]},
      ],
      questions: [
        {
          q: "Your DDP run produces different metric values across two identical runs. Why?",
          a: "Three usual suspects: (1) data sharding seeded differently, (2) NCCL non-determinism, (3) floating-point non-associativity in reductions. Sharding is 90 % of cases.",
          followups: [
            {
              q: "How do you fix sampler seeding correctly?",
              a: "<code>DistributedSampler(dataset, seed=...)</code>, plus <code>sampler.set_epoch(epoch)</code> every epoch. Forgetting <code>set_epoch</code> = silent train-set leak (rank 0 sees same shard every epoch).",
              followups: [
                {
                  q: "Could that bias eval metrics?",
                  a: "Yes — eval over a small fixed shard is higher-variance and biased. Seed val sampler deterministically and ensure each rank sees a different shard."
                }
              ]
            }
          ]
        },
        {
          q: "Why is <code>dist_sync_on_step=True</code> almost always wrong?",
          a: "Triggers a global <code>all_gather</code> inside <code>forward()</code>. Every batch becomes a global barrier — slowest rank gates the world. Throughput tanks; gradient sync compounds the cost.",
          followups: [
            {
              q: "When is it right?",
              a: "Only when per-step value must be globally consistent — e.g. a custom LR schedule reading a synced metric. Even then, prefer logging local values + framework <code>sync_dist</code> for scalars.",
              followups: [
                {
                  q: "Alternative?",
                  a: "Two metric instances: un-synced one for fast per-step logs (local rank); synced one called only periodically. Doubles state but no per-step barrier."
                }
              ]
            }
          ]
        },
        {
          q: "How does <code>_sync_dist</code> handle list states with very different sizes per rank?",
          a: "<code>gather_all_tensors</code>: <code>all_gather</code>s shapes, pads each tensor to max shape with zeros, <code>all_gather</code>s the padded tensors, slices each contribution back to its real shape.",
          followups: [
            {
              q: "Doesn't padding waste memory and bandwidth?",
              a: "Yes — proportional to shape variance. Negligible for typical workloads; meaningful for pathological cases (one rank with 10× the data).",
              followups: [
                {
                  q: "If you can't balance shards?",
                  a: "Switch to tensor-state metric (sum/mean reductions, fixed shape). For canonically list-state metrics (AUROC, mAP), use a binned variant — bucket predictions, track per-bin counts. Tensor state, no padding, slightly approximate."
                }
              ]
            }
          ]
        },
        {
          q: "Custom metric works on 1 GPU, fails silently on 4. What do you check first?",
          a: "(1) Every state is a <code>Tensor</code> or list of <code>Tensor</code> (not Python list of floats). (2) Every state has explicit <code>dist_reduce_fx</code>. (3) Reductions are associative.",
          followups: [
            {
              q: "What does 'doesn't reduce' look like in state?",
              a: "After sync, your <code>tp</code> tensor has shape <code>(world_size, ...)</code> instead of <code>(...)</code>. <code>compute</code> treats the rank dim as data — silent miscalculation. Bug is <code>add_state(..., dist_reduce_fx=None)</code>."
            }
          ]
        },
      ]
    },

    {
      id: "lightning",
      category: "Distributed & Lightning",
      title: "PyTorch Lightning Integration",
      summary: [
        { h2: "The recommended pattern" },
        { pre: "class LitClassifier(pl.LightningModule):\n    def __init__(self, n):\n        super().__init__()\n        metrics = MetricCollection({\n            'acc': Accuracy(task='multiclass', num_classes=n),\n            'f1':  F1Score(task='multiclass', num_classes=n, average='macro'),\n        })\n        self.train_metrics = metrics.clone(prefix='train/')\n        self.val_metrics   = metrics.clone(prefix='val/')\n\n    def training_step(self, batch, _):\n        x, y = batch\n        logits = self.model(x)\n        loss = F.cross_entropy(logits, y)\n        # Pass the metric module — NOT metric.compute()\n        self.log_dict(self.train_metrics(logits, y), on_step=True, on_epoch=True)\n        return loss\n\n    def validation_step(self, batch, _):\n        x, y = batch\n        self.val_metrics.update(self.model(x), y)\n\n    def on_validation_epoch_end(self):\n        self.log_dict(self.val_metrics.compute())\n        self.val_metrics.reset()" },
        { h2: "Critical rules" },
        { ul: [
          "Register metrics as <b>module attributes</b> so Lightning moves them to device.",
          "Use <code>MetricCollection</code> (a <code>ModuleDict</code>), not a plain dict.",
          "Pass the metric instance to <code>self.log</code>, never <code>metric.compute()</code>.",
          "Don't set <code>sync_dist=True</code> on a metric — it has its own DDP sync.",
          "Use <code>metrics.clone(prefix=...)</code> to keep train and val state separate.",
        ]},
      ],
      questions: [
        {
          q: "Why pass the metric to <code>self.log</code> instead of <code>metric.compute()</code>?",
          a: "Lightning detects <code>Metric</code> instances and orchestrates <code>forward → compute → reset</code> across <code>on_step</code> / <code>on_epoch</code>. Passing <code>compute()</code> gives Lightning a scalar — it can't reset, sync, or accumulate.",
          followups: [
            {
              q: "What goes wrong if you do <code>self.log('acc', metric.compute())</code> in training_step?",
              a: "Logs whatever <code>compute()</code> returned — likely the cached value from the last sync, possibly stale. State isn't reset, so next epoch's compute includes this epoch's data. Trace looks correct for ~1 epoch then goes monotonic forever.",
              followups: [
                {
                  q: "Why doesn't Lightning warn?",
                  a: "It can't tell — argument is a Tensor, indistinguishable from any other scalar log.",
                  followups: [
                    {
                      q: "How to guard against this in code review?",
                      a: "Lint rule / pre-commit hook flagging <code>.compute()</code> calls inside <code>*_step</code>. Plus a unit test asserting <code>metric._update_count == 0</code> at the start of each epoch."
                    }
                  ]
                }
              ]
            }
          ]
        },
        {
          q: "What's the actual difference between <code>on_step=True, on_epoch=True</code> and only <code>on_epoch=True</code>?",
          a: "<code>on_step=True</code> calls <code>metric.forward(...)</code> per batch and logs that value. <code>on_epoch=True</code> calls <code>metric.compute()</code> at the appropriate hook and logs that. Both → two TensorBoard series.",
          followups: [
            {
              q: "Doesn't <code>forward()</code> + <code>update()</code> double-count state?",
              a: "No — <code>forward()</code> calls <code>update</code> exactly once internally. The two-pass behavior of <code>_forward_full_state_update</code> resets and re-updates only to compute the per-batch value; global state increments once."
            }
          ]
        },
        {
          q: "<code>metrics.clone(prefix='train/')</code> — what does it actually do?",
          a: "Deep-copies the entire MetricCollection and prepends <code>'train/'</code> to every key. Each cloned metric has its own state.",
          followups: [
            {
              q: "Why deep-copy and not just rename keys?",
              a: "Different states. Same instance for train and val ⇒ <code>validation_step</code> updates contaminated by training data. Validation accuracy silently includes train.",
              followups: [
                {
                  q: "Memory cost of cloning?",
                  a: "Same as original metric state. Tensor states: trivial. List states: doubles. For 50k-sample AUROC, clone before fitting and consider <code>compute_on_cpu=True</code>."
                }
              ]
            }
          ]
        },
        {
          q: "Why register metrics as module attributes?",
          a: "So Lightning's <code>.to(device)</code> traversal moves them. Storing in a dict (<code>self.metrics = {'acc': ...}</code>) bypasses <code>nn.Module.__setattr__</code> and Lightning never sees it — device-mismatch errors follow.",
          followups: [
            {
              q: "Why does <code>MetricCollection</code> work then?",
              a: "It extends <code>nn.ModuleDict</code>, which <i>does</i> register children. Setting <code>self.metrics = MetricCollection(...)</code> is correct."
            }
          ]
        },
      ]
    },

    /* ============================================================ CUSTOM & PRODUCTION */
    {
      id: "custom-metrics",
      category: "Custom & Production",
      title: "Custom Metrics",
      summary: [
        { h2: "The 5-step recipe" },
        { ul: [
          "Inherit <code>Metric</code>.",
          "Set metadata (<code>is_differentiable</code>, <code>higher_is_better</code>, <code>full_state_update</code>).",
          "Declare state with <code>add_state</code> in <code>__init__</code>.",
          "Implement <code>update(*args, **kwargs)</code> — mutate state, no return.",
          "Implement <code>compute()</code> — pure function from state to value.",
        ]},
        { h2: "Example: weighted MAE" },
        { pre: "class WeightedMAE(Metric):\n    is_differentiable = True\n    higher_is_better  = False\n    full_state_update = False\n\n    def __init__(self, **kwargs):\n        super().__init__(**kwargs)\n        self.add_state('sum_abs_error', default=torch.tensor(0.0), dist_reduce_fx='sum')\n        self.add_state('sum_weights',   default=torch.tensor(0.0), dist_reduce_fx='sum')\n\n    def update(self, preds, target, weights):\n        abs_err = (preds - target).abs()\n        self.sum_abs_error += (abs_err * weights).sum()\n        self.sum_weights   += weights.sum()\n\n    def compute(self):\n        return self.sum_abs_error / self.sum_weights.clamp(min=1e-12)" },
        { h2: "Don'ts" },
        { ul: [
          "Don't store Python lists / floats as state — they won't sync.",
          "Don't call <code>torch.distributed</code> inside <code>update()</code>.",
          "Don't return values from <code>update()</code> — they're ignored.",
          "Don't compute in <code>update()</code> — keep math in <code>compute()</code>.",
          "Don't depend on insertion order — DDP <code>cat</code> doesn't guarantee order.",
        ]},
      ],
      questions: [
        {
          q: "You set <code>full_state_update=False</code> but <code>forward()</code> returns wrong batch values. Why?",
          a: "The fast path assumes batch state can be merged via the registered reduction. If your <code>compute</code> reads anything beyond the declared states, the merge will be wrong.",
          followups: [
            {
              q: "How do you tell?",
              a: "Run a unit test: one big update vs. two half-updates via <code>forward</code>. The two batch values must match expected per-batch results, and final <code>compute()</code> must match the single-update result.",
              followups: [
                {
                  q: "Cost of <code>full_state_update=True</code>?",
                  a: "Two <code>update()</code>s per <code>forward()</code>. Minor for cheap updates, substantial for expensive ones (BERT encoding etc.). Better fix: refactor compute to use only declared states."
                }
              ]
            }
          ]
        },
        {
          q: "Custom metric with per-sample weights — how?",
          a: "Add <code>weights</code> as a third argument to <code>update</code>. Multiply preds × weights elementwise before accumulating. Make weights default to <code>None</code>; branch internally to <code>ones_like(target)</code>.",
          followups: [
            {
              q: "Won't this break <code>MetricCollection</code> compute groups?",
              a: "If some metrics in the collection accept weights and others don't, compute groups won't share state — different signatures detected. Either separate collections, or unify all metrics to accept optional weights."
            }
          ]
        },
        {
          q: "How would you write a custom metric that does bootstrap CI internally?",
          a: "Maintain N parallel copies of states. Each <code>update</code> samples which copies the batch goes into via Poisson(1). At <code>compute</code>, you have N values — return mean and quantiles.",
          followups: [
            {
              q: "Why Poisson(1)?",
              a: "Asymptotic limit of with-replacement sampling. Each item appears in each bootstrap copy with the right multinomial expectation. Memory-efficient: store inclusion counts, not sample IDs.",
              followups: [
                {
                  q: "Isn't this just <code>BootStrapper</code>?",
                  a: "Yes — use it instead of writing your own unless you need a custom strategy (paired bootstrap, stratified bootstrap)."
                }
              ]
            }
          ]
        },
      ]
    },

    {
      id: "production",
      category: "Custom & Production",
      title: "Production Scenarios",
      summary: [
        { h2: "Patterns" },
        { ul: [
          "<b>Offline harness</b>: <code>MetricCollection</code> + <code>BootStrapper</code> for CIs.",
          "<b>Online drift</b>: <code>Running(metric, window=10000)</code>, emit on schedule.",
          "<b>A/B</b>: per-variant metrics + paired bootstrap for differences.",
          "<b>Segmented</b>: per-region/device/version metric instances.",
          "<b>Multi-task</b>: <code>MultitaskWrapper</code> + per-task collections.",
          "<b>Checkpointable state</b>: <code>add_state(..., persistent=True)</code>.",
          "<b>Fairness CI</b>: <code>BinaryFairness</code> in launch gates.",
        ]},
        { h2: "Three layers of metric, always" },
        { ul: [
          "Model-quality (TorchMetrics number).",
          "Decision-quality (operating point, dollar-loss).",
          "Outcome (online lift, revenue, NPS).",
        ]},
      ],
      questions: [
        {
          q: "Build the metric layer of a real-time recommender at 50k QPS.",
          a: "Per-region worker holds <code>Running(metric, window=...)</code> instances. Update batches of ~50ms or 1k items. Emit <code>compute()</code> every 30s to TSDB. Sharded by region; aggregate offline for global numbers.",
          followups: [
            {
              q: "Why a rolling window, not periodic reset?",
              a: "Window smooths per-batch noise. 30s tick on a 10k-event window is more stable than 30s tick on the last 30s of events alone.",
              followups: [
                {
                  q: "Right window size?",
                  a: "Cover ≥ 1k events for stability; not so long that drift is invisible. For 50k QPS, ~30k events ≈ 600ms of traffic. Scale with QPS.",
                  followups: [
                    {
                      q: "Persisting window state across restarts?",
                      a: "<code>metric.metric_state</code> is a dict of tensors — pickle on shutdown, restore on startup. Without it, every restart forgets the last window."
                    }
                  ]
                }
              ]
            }
          ]
        },
        {
          q: "Production F1 drifts down over 2 weeks. Investigation order?",
          a: "(1) Input distribution shift. (2) Label distribution shift. (3) Label-pipeline integrity (lag, schema). (4) Model staleness. (5) Infra (feature serving, deployment).",
          followups: [
            {
              q: "How to measure (1) and (2)?",
              a: "Custom metrics: <code>KLDivergence</code> between current feature batches and a reference distribution; <code>MeanMetric</code> on positive rate. Add to monitoring stack alongside F1.",
              followups: [
                {
                  q: "Alert threshold for KL?",
                  a: "Empirical: baseline KL during stable period; alert when current >3σ for sustained windows.",
                  followups: [
                    {
                      q: "Why sustained?",
                      a: "Single-point alerts are noisy (transient batch oddities). Standard SRE pattern: '5 of last 10 windows breach' avoids false pages."
                    }
                  ]
                }
              ]
            }
          ]
        },
        {
          q: "Δ F1 = 0.005 over baseline — real win?",
          a: "Run a paired bootstrap. <code>BootStrapper(F1, num_bootstraps=1000, quantile=...)</code> on both, paired on sample index. If 95% CI on the difference excludes zero, real win. If it includes zero, eval set isn't large enough.",
          followups: [
            {
              q: "Sample size to detect Δ=0.005?",
              a: "For per-sample F1 σ ~ 0.02, need n ≥ (1.96·σ/Δ)²·2 ≈ 60k for paired difference at 95% CI / 80% power. 5k eval set can't reliably detect 0.005."
            }
          ]
        },
      ]
    },

    /* ============================================================ INTERVIEW PREP */
    {
      id: "interview-quickfire",
      category: "Interview Prep",
      title: "Interview Quick-fire",
      summary: [
        { h2: "Use this page like flashcards" },
        { p: "These are the most-asked TorchMetrics questions, condensed. Switch to Quiz mode to drill them with multi-level follow-ups." },
      ],
      questions: [
        {
          q: "What's TorchMetrics and why does it exist?",
          a: "PyTorch-native library of 100+ tested metrics + a <code>Metric</code> base class for custom ones. Started inside Lightning, split out so plain-PyTorch users could share the same battle-tested implementations. Solves: non-decomposability, DDP correctness, device placement, API consistency.",
          followups: [
            {
              q: "Why can't you just average per-batch accuracy?",
              a: "Most metrics are non-decomposable when batches differ in size/balance. Average of batch accuracies ≠ correct global accuracy. Fix: aggregate sufficient statistics, then compute once."
            }
          ]
        },
        {
          q: "What happens when I call <code>metric(preds, target)</code>?",
          a: "It's <code>forward()</code>. Two paths: <code>_forward_full_state_update</code> (safe, two updates) or <code>_forward_reduce_state_update</code> (fast, one update + state merge). Result: per-batch value plus updated global state.",
          followups: [
            {
              q: "When does the fast path apply?",
              a: "<code>full_state_update=False</code> AND every state reduces by sum/mean/min/max/cat."
            }
          ]
        },
        {
          q: "How is DDP correctness achieved?",
          a: "Each state has <code>dist_reduce_fx</code>. <code>compute()</code> calls <code>_sync_dist</code>: pre-concat list states locally, <code>all_gather</code> across ranks (with shape-padding), apply reduction. Local state cached and restored on exit so subsequent updates work."
        },
        {
          q: "What's a compute group in MetricCollection?",
          a: "Metrics with the same internal state (e.g. all StatScores-derived) share one <code>update()</code> call; the rest derive from it in <code>compute()</code>. 3–10× speedup for collections of related classification metrics."
        },
        {
          q: "Tensor state vs list state?",
          a: "Tensor: O(1) memory, summable, DDP-friendly. List: keeps full population (needed for AUROC, mAP, NDCG, BLEU); grows with eval; use <code>compute_on_cpu=True</code> for huge evals."
        },
        {
          q: "Why must you call <code>reset()</code>?",
          a: "<code>compute()</code> doesn't reset. Skipping <code>reset</code> between epochs leaks data forward. Lightning calls it for you at the right hooks; raw PyTorch you do yourself."
        },
        {
          q: "What's <code>compute_on_cpu</code>?",
          a: "Moves list states to CPU after each update. Trades PCIe traffic for GPU RAM. Use for million-sample AUROC / mAP / BLEU evals."
        },
        {
          q: "<code>BootStrapper</code> — how does it work?",
          a: "Maintains N internal copies of the wrapped metric. Each gets a Poisson-bootstrapped subsample of every batch. <code>compute</code> returns mean + quantiles across replicas — non-parametric CI."
        },
        {
          q: "<code>MetricTracker</code> — what's it for?",
          a: "Wraps a metric to keep a history across epochs. Exposes 'best so far' / 'best epoch' using <code>higher_is_better</code>. Removes hand-written best-val bookkeeping."
        },
        {
          q: "<code>merge_state</code> — what's it for?",
          a: "Folds another metric's (or dict's) state into the current one, using registered reductions. Lets you persist states from different machines/pipelines and reduce offline. Requires <code>full_state_update=False</code> or override."
        },
        {
          q: "How to compute CI on F1?",
          a: "<code>BootStrapper(F1Score(...), num_bootstraps=1000, quantile=torch.tensor([0.025, 0.975]))</code>. For paired model comparison, write a paired-bootstrap variant."
        },
        {
          q: "Multilabel vs multiclass?",
          a: "Multiclass: 1 class per sample, argmax over K. Multilabel: K independent binary tasks, threshold each. New API is task-prefixed (<code>BinaryX</code>, <code>MulticlassX</code>, <code>MultilabelX</code>)."
        },
      ]
    },

    {
      id: "system-design",
      category: "Interview Prep",
      title: "System Design",
      summary: [
        { h2: "Common SD prompts you'll get" },
        { ul: [
          "Real-time evaluation service for a recommender at 50k QPS.",
          "Offline eval pipeline for a foundation-model finetune.",
          "Fairness-gated CI for model releases.",
          "Edge object detection with privacy-preserving cloud aggregation.",
          "FID-at-scale for thousands of generative-model runs.",
          "A/B-testable metric system.",
          "Compliance-grade audit ledger.",
        ]},
        { h2: "How to answer in an interview" },
        { ol: [
          "Clarify SLO, cardinality, latency budget.",
          "Show where TorchMetrics fits and where it doesn't.",
          "Call out failure modes (drift, schema change, restart).",
          "Frame trade-offs, not 'the right answer'.",
        ]},
      ],
      questions: [
        {
          q: "Design a real-time eval service for a recommender at 50k QPS.",
          a: "Sharded workers per (region, model_version). Each holds <code>Running(metric, window=10k)</code>. Updates batched ~50ms / 1k items. Emit <code>compute()</code> every 30s to TSDB. Don't sync across workers in real time; reserve global aggregation for nightly batch.",
          followups: [
            {
              q: "Why per-shard, not global?",
              a: "Single global metric is a hot lock. Shard horizontally; aggregate offline. Keeps QPS scalable.",
              followups: [
                {
                  q: "How to handle worker restarts?",
                  a: "Persist <code>metric.metric_state</code> to disk on shutdown; restore on startup. Or accept the window-size-worth of forgotten data.",
                  followups: [
                    {
                      q: "Cardinality cap on segments?",
                      a: "Yes — a new dimension every day blows up the TSDB. Limit and drop tail. Auto-rotate dashboards by traffic-rank."
                    }
                  ]
                }
              ]
            }
          ]
        },
        {
          q: "Design FID-at-scale for thousands of generative runs.",
          a: "Pre-compute Inception activation mean+cov for the reference set ONCE. Cache to shared storage. Each run loads ref stats and only updates the 'fake' branch. <code>FrechetInceptionDistance</code> already supports this via the <code>real=True/False</code> flag in update.",
          followups: [
            {
              q: "What invalidates the cache?",
              a: "Anything that changes preprocessing or feature extractor: image resolution, normalization, Inception V3 weights, dtype. Hash (preprocessing config + extractor checksum) into the cache filename.",
              followups: [
                {
                  q: "Teammate gets different FID on 'same' data — what to check first?",
                  a: "(1) Inception backbone version. (2) Preprocessing pipeline. (3) Image data range (0-1 vs 0-255). (4) Sample count (FID biased at small N). (5) TorchMetrics version."
                }
              ]
            }
          ]
        },
        {
          q: "Design a fairness-gated CI for model releases.",
          a: "Eval CI builds a <code>MetricCollection</code> with top-line + per-segment metrics. Compares to production via paired bootstrap (<code>BootStrapper</code>). Fails the PR if: top-line drops > τ, any group drops > τ_group, fairness ratios outside [0.8, 1.25].",
          followups: [
            {
              q: "Why paired bootstrap?",
              a: "Eval-set is the same for both models. Paired difference has lower variance than independent CIs.",
              followups: [
                {
                  q: "Cache the reference model's predictions?",
                  a: "Yes. Re-running the production model on every PR is wasteful. Persist preds to S3, key by model checksum + eval-set version."
                }
              ]
            }
          ]
        },
      ]
    },

    /* ============================================================ BUSINESS MAPPING */
    {
      id: "aa-business",
      category: "Business Mapping",
      title: "American Airlines",
      summary: [
        { h2: "The 4 bridges from ML to business" },
        { ul: [
          "<b>Cost-of-error matrix</b>: map TP/FP/TN/FN to dollars (DBC, IROPS, missed connections).",
          "<b>Top-k truncation</b>: only K options shown (rebooker offers).",
          "<b>Threshold operating point</b>: precision/recall tied to ops cost.",
          "<b>Calibration</b>: probability used directly downstream (pricing, gate planning).",
        ]},
        { h2: "Use cases × metrics" },
        { table: [
          ["Use case", "Primary metrics"],
          ["Delay prediction", "BinaryF1, AUROC, ECE, Recall@FixedPrecision per hub"],
          ["No-show / overbooking", "Quantile loss (custom), CRPS, coverage"],
          ["Dynamic pricing", "Brier (MSE on probs), ECE, Demand MAE"],
          ["IROPS rebooker", "RetrievalNDCG, RetrievalMRR, Recall@5"],
          ["Customer churn", "BinaryAveragePrecision, time-discounted recall"],
          ["Demand forecasting", "wMAPE, SMAPE, NRMSE"],
        ]},
      ],
      questions: [
        {
          q: "Flight delay prediction — why F1 and not just accuracy?",
          a: "~80% of flights are on-time. Accuracy of 0.80 is the trivial 'predict on-time always' baseline. F1 forces both precision and recall on the rare delay class.",
          followups: [
            {
              q: "Why not just maximize recall (FN cost is higher)?",
              a: "Over-recall floods ops with false alarms; trust erodes; ops ignore the model. Right answer: <code>RecallAtFixedPrecision(min_precision=0.85)</code>.",
              followups: [
                {
                  q: "If the precision constraint can't be met?",
                  a: "Either feature set is insufficient (add weather, upstream-delay propagation) or threshold is too aggressive. Stopgap: two operating points — high-confidence auto-action, medium-confidence monitoring queue.",
                  followups: [
                    {
                      q: "How do you measure the value of the monitor queue?",
                      a: "A/B test against historical no-action data. Right metric: <i>dollar-weighted lift</i> = cost_avoided_with_action − cost_avoided_no_action. Custom metric, sum-reduced, accumulates dollars saved."
                    }
                  ]
                }
              ]
            }
          ]
        },
        {
          q: "No-show prediction — why quantile loss instead of MSE?",
          a: "MSE forces the conditional mean. The mean isn't the operating point. Cost of being wrong on the high side (DBC) far exceeds the low side (one empty seat). You want the upper tail; quantile loss gives it.",
          followups: [
            {
              q: "What gradient flow?",
              a: "Pinball loss at quantile τ: gradient is −τ if y > ŷ, (1−τ) if y < ŷ. Asymmetry is the whole point.",
              followups: [
                {
                  q: "TorchMetrics doesn't ship pinball loss. How to add?",
                  a: "Subclass <code>Metric</code>. Two states: <code>sum_loss</code> and <code>n</code>, both sum-reduced. <code>update</code> computes pinball, accumulates. <code>compute</code> divides. ~15 lines including DDP.",
                  followups: [
                    {
                      q: "Multiple quantiles in one pass?",
                      a: "Make τ a length-K tensor. Both states become K-dim. Sum reduction works elementwise. Wrap with <code>ClasswiseWrapper</code> to label outputs."
                    }
                  ]
                }
              ]
            }
          ]
        },
        {
          q: "Dynamic pricing — why is calibration the right primary metric?",
          a: "Optimizer takes E[revenue] = price × P(book at price). 10% biased P(book) ⇒ 10% biased revenue forecast ⇒ wrong price. Calibration directly measures the bias.",
          followups: [
            {
              q: "Will better calibration always mean more revenue?",
              a: "No. A calibrated but low-resolution model — same P everywhere — is useless even at perfect calibration. You need calibration AND discrimination.",
              followups: [
                {
                  q: "Measure that trade-off?",
                  a: "Brier score decomposition: Brier = uncertainty − resolution + reliability. Want low reliability (calibrated) AND high resolution (discriminating).",
                  followups: [
                    {
                      q: "Implement it?",
                      a: "Custom metric with three accumulators: <code>bin_count</code>, <code>bin_pos_sum</code>, <code>bin_pred_sum</code>. After accumulation: per-bin observed and predicted rates → reliability; bin-mean spread → resolution. ~40 lines."
                    }
                  ]
                }
              ]
            }
          ]
        },
        {
          q: "IROPS rebooker — why NDCG, not classification accuracy?",
          a: "Multiple 'right' itineraries exist (any acceptable resolves the disruption). Being slightly off — chosen at rank 2 vs 1 — is much less bad than rank 8. NDCG captures that gradient.",
          followups: [
            {
              q: "Graded preferences (e.g. business-class prefers same-cabin 5×)?",
              a: "Use graded NDCG: target is relevance grade, not 0/1. <code>RetrievalNormalizedDCG</code> accepts non-binary targets via gain formula <code>(2^rel - 1)</code>.",
              followups: [
                {
                  q: "How to set grades without label leakage?",
                  a: "Grades come from <i>deterministic business rules</i> — fare class, cabin, FF status, original arrival time. Anything derived from 'would they accept this offer' is a leak."
                }
              ]
            }
          ]
        },
        {
          q: "Customer churn — AUROC vs AP for a rare event?",
          a: "AP penalizes FPs in proportion to recall (PR-curve area). AUROC integrates over FPR — dominated by negatives, can stay high with bad PR. <b>Use AP primary, AUROC diagnostic.</b>",
          followups: [
            {
              q: "Threshold for retention offers?",
              a: "Cost-driven: offer cost ≈ $50, LTV ≈ $400. Fire when P(churn) × LTV > offer_cost / lift_rate. At typical lift_rate=30%, threshold ~ P(churn) > 0.15.",
              followups: [
                {
                  q: "Estimating lift_rate?",
                  a: "Holdout: randomly don't-offer 5% of triggered users. Compare retention 6 months later. Lift = (retention_offered − retention_holdout) / retention_holdout. The holdout is the bridge from ML to business."
                }
              ]
            }
          ]
        },
        {
          q: "Demand forecasting — why wMAPE not MAPE?",
          a: "MAPE divides by y, exploding for low-volume routes (charters, cruises = 'infinity error'). wMAPE divides total error by total actual — robust on low volume, still scale-free.",
          followups: [
            {
              q: "Why not RMSE?",
              a: "RMSE in passenger-units doesn't compare across routes. Cross-route comparability is critical for planning prioritization."
            }
          ]
        },
      ]
    },

    {
      id: "amazon-business",
      category: "Business Mapping",
      title: "Amazon",
      summary: [
        { h2: "Workloads × metrics" },
        { table: [
          ["Workload", "Primary metrics"],
          ["Search ranking", "Recall@1000, NDCG@10, MRR, Precision@5"],
          ["Recommendations", "NDCG, HitRate, Recall@k + online lift"],
          ["Fraud / payment risk", "Recall@FixedPrecision, AP, ECE, dollar-loss"],
          ["Demand forecasting (SKU×FC×day)", "wMAPE, CRPS, quantile loss"],
          ["Delivery time", "Asymmetric quantile loss, on-time rate per lane"],
          ["Ad ranking", "ECE, NDCG@5, RPM lift"],
          ["Alexa / voice", "WER, content-WER, Slot-F1, task-completion"],
        ]},
        { h2: "Two-stage retrieval architecture" },
        { p: "First stage: Recall@1000 evaluates candidate generation. Reranker: NDCG@10 evaluates ordering quality. Different roles, different metrics." },
      ],
      questions: [
        {
          q: "Why two-stage retrieval with two different metrics?",
          a: "First-stage candidate generation must <i>not lose</i> a relevant item — recall is brutal on miss. Reranker has the recalled items in hand and is judged on <i>ordering</i> of the top.",
          followups: [
            {
              q: "Why recall@1000 specifically?",
              a: "Catalog is hundreds of millions; reranker can chew ~1k in latency budget (~50ms). K is set by reranker capacity, not by relevance preference.",
              followups: [
                {
                  q: "Ground-truth for recall@1000?",
                  a: "Pooled human-labeled judgments (slow, expensive, definitive) + click/purchase logs (cheap, biased). Production uses both: humans for canonical eval, logs for A/B.",
                  followups: [
                    {
                      q: "Correcting position bias in click logs?",
                      a: "Inverse-propensity-scoring (IPS): each click reweighted by 1 / P(item shown at that position). Estimate propensities via interleaving experiments. Metric becomes 'IPS-NDCG'."
                    }
                  ]
                }
              ]
            }
          ]
        },
        {
          q: "Recommendations — why offline metrics fail to predict online lift?",
          a: "Distribution shift. Offline NDCG measures rank quality on logged data — biased by the prior recommender. A new recommender changes what users see, which changes what they click, which changes the distribution.",
          followups: [
            {
              q: "How to make offline more predictive?",
              a: "(a) Counterfactual eval with IPS. (b) Simulation: build user-response model and replay candidate recommenders. (c) Reserve traffic for uniform-random exploration to get unbiased eval set.",
              followups: [
                {
                  q: "Trade-offs of (c)?",
                  a: "Costs short-term revenue (sometimes-bad recs). Benefit: unbiased eval, ship faster with confidence. Most large platforms run 1–5% exploration."
                }
              ]
            }
          ]
        },
        {
          q: "Fraud — why precision-at-fixed-recall, not F1?",
          a: "Fraud has hard precision constraint: cannot block more than X% of legit orders without unacceptable customer-experience damage. F1 trades precision for recall at 1:1, which doesn't reflect business cost.",
          followups: [
            {
              q: "Why 99% precision specifically?",
              a: "Most retailers operate at 99–99.5% legit-pass-through. Below that, customer-complaint volume overwhelms support regardless of fraud-catch rate.",
              followups: [
                {
                  q: "If the model can't hit 99% at usable recall?",
                  a: "Two paths: (1) serial review — model flags top X% for manual review. (2) Tiered models — high-precision auto-block + high-recall friction (3DS challenge).",
                  followups: [
                    {
                      q: "Measuring the value of friction?",
                      a: "(abandoned_legit × cart_value × abandon_repeat_rate − catched_fraud × chargeback_value) / total_friction_events. The number the friction team optimizes."
                    }
                  ]
                }
              ]
            }
          ]
        },
        {
          q: "Demand forecasting — why P90 and not the mean forecast?",
          a: "Cost asymmetry: stockouts cost more than overstock. Holding 90th-percentile demand means stockout in only 10% of weeks — typical service-level target.",
          followups: [
            {
              q: "Different SKUs have different cost asymmetries — how reflect?",
              a: "Per-SKU quantile. Newsvendor: q* = cost_underage / (cost_underage + cost_overage). For a $100 SKU with $5 holding and $30 lost-margin: q* = 30/35 = 0.857 → P85.",
              followups: [
                {
                  q: "Evaluate per-SKU quantile coverage in production?",
                  a: "Custom metric: per-SKU fraction of weeks where actual ≤ predicted_quantile. Aggregate: coverage-error = mean |hits/n − target_q|. State per-SKU (hits, n), summed."
                }
              ]
            }
          ]
        },
        {
          q: "Ad ranking — why is calibration mission-critical?",
          a: "Auction price = bid × P(click). 10% miscalibrated P(click) ⇒ 10% wrong auction price ⇒ broken pacing for every advertiser. Discrimination (AUC) without calibration is worthless.",
          followups: [
            {
              q: "Biases in P(click)?",
              a: "Position bias (top slots clicked because they're top), selection bias (only saw clicks on shown ads). Both must be debiased before calibration is meaningful.",
              followups: [
                {
                  q: "How does calibration error compose with debiasing?",
                  a: "Sequential. First debias logs (IPS / counterfactual reweighting); then evaluate calibration on debiased data. Weight the <code>update(...)</code> arguments — write a small custom IPS wrapper."
                }
              ]
            }
          ]
        },
        {
          q: "Alexa — why doesn't lower WER always mean higher task completion?",
          a: "Some words matter more than others. Misrecognizing 'play' as 'pay' breaks a music command but is invisible to overall WER. Relevant signal: content-word WER (errors on words NLU actually uses).",
          followups: [
            {
              q: "How to compute content-word WER?",
              a: "Tag each reference word as content/function via NLU schema. Run WER per tag class. <code>WordErrorRate</code> wrapped in <code>MultitaskWrapper</code>.",
              followups: [
                {
                  q: "Tie back to dialog success?",
                  a: "Bucket dialogs by content-WER; measure task-completion per bucket. Slope dTaskCompletion/dContentWER is the effective business gradient — that's what you optimize."
                }
              ]
            }
          ]
        },
      ]
    },

    {
      id: "scenarios",
      category: "Business Mapping",
      title: "Scenario Setups",
      summary: [
        { h2: "Three layers of metric, every scenario" },
        { ol: [
          "Fast training-loop metric (single number every step).",
          "Broad validation MetricCollection (the dashboard).",
          "Segmented breakdown (per-class, per-region, per-version).",
        ]},
        { h2: "10 canonical setups" },
        { ul: [
          "ImageNet classification (Top-1/Top-5/F1/ECE + per-class)",
          "Object detection (mAP, compute_on_cpu=True)",
          "Medical segmentation (Dice + IoU + Hausdorff)",
          "Generative image (FID + KID + LPIPS + CLIPScore)",
          "LLM eval (Perplexity + ROUGE + BERTScore)",
          "Recommender (offline NDCG + online lift)",
          "ASR (WER + CER + per-domain)",
          "Time-series forecasting (wMAPE + CRPS + per-level)",
          "Anomaly detection (AP + AUROC + Recall@FixedPrecision)",
          "Multi-task models (MultitaskWrapper)",
        ]},
      ],
      questions: [
        {
          q: "Image classification training — why log both <code>on_step=True</code> and <code>on_epoch=True</code>?",
          a: "Step series shows training noise / instability mid-epoch (useful for LR-tuning). Epoch series gives clean compare-across-epochs trend. Two TensorBoard series; near-zero overhead because step logs are local-rank.",
          followups: [
            {
              q: "Doesn't <code>on_step=True</code> add DDP barriers?",
              a: "No. Lightning + TorchMetrics does NOT all_gather on <code>forward()</code> by default. Sync only at <code>compute()</code>. Step logging is local-rank, near-zero overhead.",
              followups: [
                {
                  q: "I want a step-level globally synced metric for debugging?",
                  a: "Set <code>dist_sync_on_step=True</code>. Costs a global barrier per step. OK for short debug runs, not for prod training.",
                  followups: [
                    {
                      q: "Avoid the barrier but still get a quick global view?",
                      a: "Lightning's <code>sync_dist=True</code> on a scalar loss — syncs the reduced number, not metric state. Different mechanism, much cheaper, only valid for sums/means."
                    }
                  ]
                }
              ]
            }
          ]
        },
        {
          q: "Object detection — why no GPU mAP in TorchMetrics?",
          a: "COCO evaluator runs in NumPy/Cython on CPU. Sorting + Hungarian matching is awkward to fuse on GPU. TorchMetrics ships the canonical implementation for trust.",
          followups: [
            {
              q: "Need fast eval during training?",
              a: "Cheaper proxy: per-class top-1 on classifier head + box-IoU on regression head. Standard tensor-state metrics. Run real mAP every N val epochs.",
              followups: [
                {
                  q: "Won't proxies drift from real mAP?",
                  a: "Track correlation; recalibrate proxy if correlation drops below 0.9. Below 0.9, proxy may pick worse models.",
                  followups: [
                    {
                      q: "Why 0.9?",
                      a: "Empirical: at correlation < 0.9 the proxy starts picking different winners than full mAP. Below that, you risk shipping a worse model."
                    }
                  ]
                }
              ]
            }
          ]
        },
        {
          q: "LLM eval — why both ROUGE and BERTScore?",
          a: "ROUGE is lexical-overlap — quick, deterministic, sensitive to surface form. BERTScore captures semantic similarity but is slower and depends on a reference model. Together: 'did you use the right words' AND 'did you convey the right meaning'.",
          followups: [
            {
              q: "Optimize ROUGE directly via RL?",
              a: "Possible (SCST), but gives short-term wins on the metric while degrading human preference. Optimize cross-entropy; treat ROUGE/BERTScore as eval, not training signal.",
              followups: [
                {
                  q: "What's the ground truth for an LLM?",
                  a: "Human preference (SBS, ELO). Automated metrics are proxies. Pipeline: cross-entropy (every step) → ROUGE/BERTScore/MMLU (every checkpoint) → human SBS (only on shortlist)."
                }
              ]
            }
          ]
        },
        {
          q: "Medical segmentation — why Dice <i>and</i> IoU?",
          a: "Monotonic but not affine. Dice rewards near-perfect overlap more steeply. Reviewers expect both. Hausdorff for boundary quality.",
          followups: [
            {
              q: "Hausdorff is brittle — alternatives?",
              a: "It's worst-case boundary error; one rogue voxel dominates. For clinical reporting, prefer ASSD (average symmetric surface distance) or HD95 (95th-percentile Hausdorff).",
              followups: [
                {
                  q: "Add HD95 to TorchMetrics?",
                  a: "Subclass <code>HausdorffDistance</code>. Override <code>compute()</code> to take 95th percentile of per-voxel distance distribution rather than max. List state of pairwise distances, cat reduction, percentile in compute."
                }
              ]
            }
          ]
        },
        {
          q: "Generative image eval — why pre-feed real images?",
          a: "FID's real statistics depend only on the real distribution and feature extractor. Recomputing per eval wastes 50k Inception forwards. Cache once.",
          followups: [
            {
              q: "How to cache practically?",
              a: "Persist <code>metric.metric_state</code> after the real-feed loop. Next run: <code>metric.merge_state(loaded_state)</code> restores running mean/cov. Both states are tensor-state with sum reduction → DDP-friendly."
            }
          ]
        },
      ]
    },

    /* ============================================================ REFERENCE */
    {
      id: "troubleshooting",
      category: "Reference",
      title: "Troubleshooting",
      summary: [
        { h2: "Quick triage table" },
        { table: [
          ["Symptom", "Most likely cause / first action"],
          ["Wrong number, single GPU", "<code>update</code> math, then <code>compute</code> math."],
          ["Wrong number, DDP only", "<code>dist_reduce_fx</code> for every state."],
          ["Hangs in DDP", "Different ranks took different paths in <code>update</code>."],
          ["OOM in eval", "List-state metric → <code>compute_on_cpu=True</code>."],
          ["Drifting values per epoch", "Forgot <code>reset()</code> or kept references."],
          ["Lightning <code>self.log</code> weirdness", "Probably passed <code>metric.compute()</code> instead of <code>metric</code>."],
          ["NaN per-class metrics", "Class has zero support; check <code>zero_division</code>."],
          ["Different result on retry", "Sampler not seeded; floating-point reduction order."],
        ]},
        { h2: "Common bugs cheat sheet" },
        { ul: [
          "<b>Same metric in train and val</b> — phases mix. Use <code>metrics.clone(prefix='train/')</code>.",
          "<b>Storing metric in dict</b> — Lightning never sees it. Use <code>MetricCollection</code> (a <code>ModuleDict</code>).",
          "<b><code>self.log('acc', metric.compute())</code></b> — pass the metric instance instead.",
          "<b>compute() before update()</b> — warns; check exception handlers around update.",
          "<b>FID inconsistency</b> — pin Inception weights, sample count, preprocessing.",
        ]},
      ],
      questions: [
        {
          q: "Metric is silently wrong only on multi-GPU. What do you check?",
          a: "(1) State is Tensor / list-of-Tensor (not Python list of floats). (2) Every <code>add_state</code> has a <code>dist_reduce_fx</code>. (3) Reductions are associative.",
          followups: [
            {
              q: "Diagnostic test?",
              a: "<code>world_size=2</code> test: each rank gets half the data. DDP value must equal single-process value over full data. If not, one of the three above is broken."
            }
          ]
        },
        {
          q: "Metric value drifts every epoch despite calling <code>reset()</code>.",
          a: "You're holding a reference to a list state and modifying it after <code>reset()</code> overwrote <code>self.&lt;name&gt;</code>. Reset replaces the attribute, doesn't truncate the old list.",
          followups: [
            {
              q: "Fix?",
              a: "Don't keep references across reset. If you need a copy, <code>copy.deepcopy(metric.metric_state)</code> before <code>reset()</code>."
            }
          ]
        },
        {
          q: "Memory blows up during validation.",
          a: "List-state metric (AUROC / mAP / BLEU / NDCG) holds all preds + targets until compute. Pass <code>compute_on_cpu=True</code> at construction. Consider chunking eval into shards and merging via <code>merge_state</code>.",
          followups: [
            {
              q: "Still OOM?",
              a: "Switch to a tensor-state approximation (binned AUROC) or shard eval and process offline. List-state is fundamentally O(N)."
            }
          ]
        },
      ]
    },
  ]
};
