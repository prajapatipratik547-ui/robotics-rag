"""Step 4 of fine-tuning: did it help? Fine-tuned vs base embedding model.

Reads the per-question ranks that eval/run_eval.py wrote for each model
(eval/finetune/<model>.json) and reports, with paired 95% bootstrap intervals:

1. In-corpus: the model fine-tuned on queries for all documents, on all 85
   questions. This is the realistic setting (a RAG system always has its
   corpus), but the model has seen synthetic queries for the test passages.
2. Held-out documents: each question is scored by the model trained without
   its fold's documents, so it never saw those passages. If the gain survives
   here, the model learned the domain, not the passages.

    python train/compare.py      # writes eval/finetune/results.md + .json

The held-out rows appear for an attempt once its two fold models
(<name>-noA, <name>-noB) have been evaluated.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from folds import FOLDS, fold_question_ids  # noqa: E402

RESULTS_DIR = ROOT / "eval" / "finetune"
BASE = "bge-small-base"
# Each fine-tuning attempt: its name prefix and where its training queries came from.
ATTEMPTS = {
    "bge-small-robotics": "doc2query (train/make_queries.py)",
    "bge-small-robotics-llm": "LLM-written questions (train/make_llm_queries.py)",
}
MODES = ["dense", "hybrid", "hybrid+rerank"]
SAMPLES = 10_000


def ranks(model: str) -> dict[str, dict[str, int | None]]:
    """question id -> mode -> rank of the first relevant chunk (None = miss)."""
    data = json.loads((RESULTS_DIR / f"{model}.json").read_text(encoding="utf-8"))
    return {q["id"]: q["first_rank"] for q in data["per_question"]}


def scores(rank: int | None) -> tuple[float, float]:
    return (1.0 if rank else 0.0), (1 / rank if rank else 0.0)


def paired(base: list[int | None], tuned: list[int | None]) -> dict:
    """Means and the paired bootstrap interval of tuned - base, for Hit@5 and MRR."""
    b = np.array([scores(r) for r in base])
    t = np.array([scores(r) for r in tuned])
    draws = np.random.default_rng(0).integers(0, len(b), size=(SAMPLES, len(b)))
    out = {"n": len(b)}
    for col, metric in enumerate(["hit", "mrr"]):
        diff = t[:, col] - b[:, col]
        boot = diff[draws].mean(axis=1)
        out[metric] = {
            "base": float(b[:, col].mean()), "tuned": float(t[:, col].mean()), "diff": float(diff.mean()),
            "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        }
    return out


def row(label: str, mode: str, r: dict) -> str:
    hit, mrr = r["hit"], r["mrr"]
    if any(m["ci"][0] > 0 for m in (hit, mrr)):
        verdict = "better"
    elif any(m["ci"][1] < 0 for m in (hit, mrr)):
        verdict = "worse"
    else:
        verdict = "no clear change"
    return (
        f"| {label} | {mode} | {r['n']} | {hit['base']:.0%} → {hit['tuned']:.0%} | "
        f"{hit['ci'][0] * 100:+.0f} to {hit['ci'][1] * 100:+.0f} | {mrr['base']:.3f} → {mrr['tuned']:.3f} | "
        f"{mrr['ci'][0]:+.3f} to {mrr['ci'][1]:+.3f} | {verdict} |"
    )


def compare(name: str, base: dict) -> tuple[dict, list[str]]:
    """In-corpus and (if its fold models exist) held-out results for one attempt."""
    ids_all = list(base)
    tuned_all = ranks(name)
    results = {"in_corpus": {}, "held_out": {}}
    lines = []
    for mode in MODES:
        results["in_corpus"][mode] = paired([base[q][mode] for q in ids_all], [tuned_all[q][mode] for q in ids_all])
        lines.append(row("In-corpus (all documents)", mode, results["in_corpus"][mode]))
    if all((RESULTS_DIR / f"{name}-no{fold}.json").exists() for fold in FOLDS):
        held_out = {qid: ranks(f"{name}-no{fold}")[qid] for fold in FOLDS for qid in fold_question_ids(fold)}
        ids_held = [qid for qid in ids_all if qid in held_out]
        for mode in MODES:
            results["held_out"][mode] = paired([base[q][mode] for q in ids_held], [held_out[q][mode] for q in ids_held])
            lines.append(row("Held-out documents", mode, results["held_out"][mode]))
    return results, lines


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # the table uses "Δ" and "→"
    base = ranks(BASE)
    lines = [
        "# Fine-tuned embedding model vs base",
        "",
        "Written by `python train/compare.py` from the retrieval runs in this folder. Base: "
        f"`{BASE}` (BAAI/bge-small-en-v1.5 exported unchanged). Paired 95% bootstrap intervals "
        f"({SAMPLES:,} resamples) of fine-tuned minus base; a verdict of better or worse means an interval excludes 0.",
    ]
    results = {}
    for name, source in ATTEMPTS.items():
        if not (RESULTS_DIR / f"{name}.json").exists():
            continue
        results[name], rows = compare(name, base)
        lines += [
            "",
            f"## `{name}`: trained on {source}",
            "",
            "| Test | Mode | Questions | Hit@5 | Δ 95% CI (pts) | MRR | Δ 95% CI | Verdict |",
            "|---|---|---|---|---|---|---|---|",
            *rows,
        ]
    lines += [
        "",
        "Held-out: each question whose evidence lies in one fold of documents is scored by the model "
        "trained without that fold (train/folds.py), so it never saw synthetic questions about those "
        "passages; the 3 questions spanning both folds are left out.",
    ]
    (RESULTS_DIR / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
