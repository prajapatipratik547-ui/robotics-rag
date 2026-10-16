# Fine-tuned embedding model vs base

Written by `python train/compare.py` from the retrieval runs in this folder. Base: `bge-small-base` (BAAI/bge-small-en-v1.5 exported unchanged). Paired 95% bootstrap intervals (10,000 resamples) of fine-tuned minus base; a verdict of better or worse means an interval excludes 0.

## `bge-small-robotics`: trained on doc2query (train/make_queries.py)

| Test | Mode | Questions | Hit@5 | Δ 95% CI (pts) | MRR | Δ 95% CI | Verdict |
|---|---|---|---|---|---|---|---|
| In-corpus (all documents) | dense | 85 | 84% → 82% | -5 to +2 | 0.695 → 0.676 | -0.067 to +0.029 | no clear change |
| In-corpus (all documents) | hybrid | 85 | 95% → 93% | -6 to +0 | 0.804 → 0.766 | -0.074 to -0.005 | worse |
| In-corpus (all documents) | hybrid+rerank | 85 | 96% → 96% | +0 to +0 | 0.844 → 0.844 | +0.000 to +0.000 | no clear change |

Held-out: each question whose evidence lies in one fold of documents is scored by the model trained without that fold (train/folds.py), so it never saw synthetic questions about those passages; the 3 questions spanning both folds are left out.
