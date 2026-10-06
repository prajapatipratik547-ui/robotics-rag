"""Stage 5 evaluation: dense vs hybrid vs hybrid+rerank on a hand-built test set.

Two kinds of metrics:

1. Retrieval (no LLM: deterministic, free, takes seconds). Each test question
   names an exact phrase from the document that answers it; every chunk of that
   document containing the phrase counts as a relevant chunk. For the top 5
   chunks each mode returns:
   - Hit@5: is at least one relevant chunk in the top 5? This is the headline
     "retrieval accuracy": if no relevant chunk reaches the LLM, it can't answer.
   - MRR: 1 / rank of the first relevant chunk (1.0 = always ranked first).
   - Context recall (ragas, ID-based): share of the relevant chunks retrieved.
   - Context precision (ragas, ID-based): share of retrieved chunks that are relevant.

2. Answer quality (LLM). The generator answers from each mode's top 5 chunks,
   then a different LLM (the judge) scores the answer with ragas:
   - Faithfulness: share of the answer's claims supported by the retrieved chunks.
   - Answer accuracy: does the answer match the reference answer (0, 0.5 or 1)?

   Only hybrid+rerank (the full pipeline) gets answer metrics: the free judge
   quota (200K tokens/day) covers ~25 questions a day, so all three modes
   would take over a week. The modes are compared on the retrieval metrics.

LLM answers and judge scores are cached in eval/cache/, so a run stopped by a
rate limit resumes where it left off, and an unchanged rerun costs nothing.

With a few dozen questions, one question is several points of Hit@5, so the
report gives 95% bootstrap confidence intervals: for each mode's Hit@5 and MRR,
and paired (same resampled questions for both modes) for the differences
between modes. A difference whose interval excludes 0 is unlikely to be noise.

Usage:
    python eval/run_eval.py                    # retrieval + answer metrics
    python eval/run_eval.py --retrieval-only   # retrieval metrics only, no LLM calls
"""

import argparse
import asyncio
import hashlib
import json
import math
import sys
import time
import warnings
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import openai
from openai import AsyncOpenAI
from ragas import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerAccuracy, Faithfulness

# ragas 0.4.3 warns that this import moved to ragas.metrics.collections, but
# the ID-based metrics don't exist there yet, so this is still the only path.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall

import config
from src.generate import NOT_FOUND, SYSTEM_PROMPT, QuotaExceeded, generate
from src.retrieve import close_client, retrieve
from src.testset import CHUNKS_PATH, TESTSET_PATH, load_jsonl, relevant_chunk_ids

EVAL_DIR = ROOT / "eval"
RESULTS_PATH = EVAL_DIR / "results.md"
CACHE_PATH = EVAL_DIR / "cache" / "llm_cache.json"

# label -> (retrieval mode, rerank on/off)
CONFIGS = {
    "dense": ("dense", False),
    "hybrid": ("hybrid", False),
    "hybrid+rerank": ("hybrid", True),
}
LLM_CONFIGS = ["hybrid+rerank"]  # see the docstring: the judge quota allows one mode
RETRIEVAL_METRICS = ["hit", "mrr", "recall", "precision"]
LLM_METRICS = ["faithfulness", "accuracy"]
CI_METRICS = ["hit", "mrr"]
# Mode differences worth testing: what keyword search adds, then what reranking adds.
COMPARISONS = [("hybrid", "dense"), ("hybrid+rerank", "hybrid"), ("hybrid+rerank", "dense")]
BOOTSTRAP_SAMPLES = 10_000
METRIC_NAMES = {
    "hit": "Hit@5",
    "mrr": "MRR",
    "recall": "Context recall",
    "precision": "Context precision",
    "faithfulness": "Faithfulness",
    "accuracy": "Answer accuracy",
}



# --- retrieval metrics ---------------------------------------------------

PRECISION = IDBasedContextPrecision()
RECALL = IDBasedContextRecall()


def score_retrieval(retrieved: list[str], relevant: list[str]) -> dict:
    ranks = [rank for rank, chunk_id in enumerate(retrieved, start=1) if chunk_id in relevant]
    sample = SingleTurnSample(retrieved_context_ids=retrieved, reference_context_ids=relevant)
    return {
        "first_rank": ranks[0] if ranks else None,
        "hit": 1.0 if ranks else 0.0,
        "mrr": 1 / ranks[0] if ranks else 0.0,
        "recall": RECALL.single_turn_score(sample),
        "precision": PRECISION.single_turn_score(sample),
    }


def run_retrieval(testset: list[dict], chunks: list[dict]) -> dict:
    """results[label][question id] = {"hits": [...], "latency": s, **metrics}"""
    results: dict = {}
    for label, (mode, use_rerank) in CONFIGS.items():
        retrieve("warm-up", config.TOP_K, mode, use_rerank)  # load models before timing
        results[label] = {}
        for item in testset:
            start = time.perf_counter()
            hits = retrieve(item["question"], config.TOP_K, mode, use_rerank)
            latency = time.perf_counter() - start
            retrieved = [hit.chunk_id for hit in hits]
            scores = score_retrieval(retrieved, relevant_chunk_ids(item, chunks))
            results[label][item["id"]] = {"hits": hits, "latency": latency, **scores}
        print(f"retrieval done: {label}")
    return results


# --- answer metrics (LLM) ------------------------------------------------


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")


def cache_key(*parts: str) -> str:
    """A key that changes whenever any input changes (model, question,
    retrieved chunks, answer...), so stale cached results are never reused."""
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()


async def cached_score(cache: dict, key: str, compute) -> float:
    """The cached score for key, or compute() it. NaN means the judge failed
    (ragas turns errors such as rate limits into NaN), so it is not cached
    and gets retried on the next run."""
    if key not in cache:
        value = (await compute()).value
        if math.isnan(value):
            return value
        cache[key] = value
        save_cache(cache)
    return cache[key]


def make_judge():
    """The ragas judge: Groq's OpenAI-compatible endpoint. The SDK retries
    429s, waiting as long as Groq's retry-after header says."""
    client = AsyncOpenAI(base_url=config.GROQ_BASE_URL, api_key=config.GROQ_API_KEY, max_retries=8)
    return llm_factory(
        config.JUDGE_MODEL, client=client, temperature=0, max_tokens=4096, reasoning_effort="low"
    )


async def run_answers(testset: list[dict], retrieval: dict, cache: dict) -> bool:
    """Fill retrieval[label][qid] with answer, faithfulness and accuracy.
    Returns False if a rate limit stopped the run early."""
    judge = make_judge()
    faithfulness = Faithfulness(llm=judge)
    accuracy = AnswerAccuracy(llm=judge)
    try:
        for label in LLM_CONFIGS:
            for item in testset:
                row = retrieval[label][item["id"]]
                hits = row["hits"]
                contexts = [hit.text for hit in hits]

                key = cache_key(
                    "answer", config.GROQ_MODEL, SYSTEM_PROMPT, item["question"], *(h.chunk_id for h in hits)
                )
                if key not in cache:
                    cache[key] = generate(item["question"], hits).text
                    save_cache(cache)
                answer = row["answer"] = cache[key]

                # A "not found" reply makes no claims, so faithfulness doesn't apply.
                if answer != NOT_FOUND:
                    row["faithfulness"] = await cached_score(
                        cache,
                        cache_key("faithfulness", config.JUDGE_MODEL, item["question"], answer, *contexts),
                        lambda: faithfulness.ascore(
                            user_input=item["question"], response=answer, retrieved_contexts=contexts
                        ),
                    )
                row["accuracy"] = await cached_score(
                    cache,
                    cache_key("accuracy", config.JUDGE_MODEL, item["question"], answer, item["reference"]),
                    lambda: accuracy.ascore(
                        user_input=item["question"], response=answer, reference=item["reference"]
                    ),
                )
                print(
                    f"  {label:<14} {item['id']}  faithfulness={fmt(row.get('faithfulness'))}"
                    f"  accuracy={fmt(row['accuracy'])}"
                )
    except (openai.RateLimitError, QuotaExceeded) as e:
        print(f"\nStopped early by a rate limit: {e}\nProgress is cached; rerun later to resume.")
        return False
    return True


# --- reporting -----------------------------------------------------------


def fmt(value) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.2f}"


def mean(rows: list[dict], metric: str) -> tuple[float | None, int]:
    """Mean over the rows that have a usable score, and how many that was."""
    values = [r[metric] for r in rows if r.get(metric) is not None and not math.isnan(r[metric])]
    return (sum(values) / len(values) if values else None), len(values)


def resample(n: int) -> np.ndarray:
    """BOOTSTRAP_SAMPLES rows of n question indices drawn with replacement. A fixed
    seed keeps the report reproducible; every interval uses the same draws."""
    return np.random.default_rng(0).integers(0, n, size=(BOOTSTRAP_SAMPLES, n))


def interval(samples: np.ndarray) -> list[float]:
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def confidence_intervals(retrieval: dict, testset: list[dict]) -> dict:
    """95% bootstrap intervals: per mode, and paired for the mode differences."""
    ids = [item["id"] for item in testset]
    draws = resample(len(ids))
    scores = {
        label: {m: np.array([retrieval[label][qid][m] for qid in ids]) for m in CI_METRICS} for label in CONFIGS
    }
    modes = {label: {m: interval(scores[label][m][draws].mean(axis=1)) for m in CI_METRICS} for label in CONFIGS}
    differences = []
    for a, b in COMPARISONS:
        row = {"modes": [a, b]}
        for m in CI_METRICS:
            diff = scores[a][m] - scores[b][m]
            row[m] = {"diff": float(diff.mean()), "ci": interval(diff[draws].mean(axis=1))}
        differences.append(row)
    return {"modes": modes, "differences": differences}


def fmt_score(value: float, metric: str, signed: bool = False) -> str:
    """Hit@5 as a percentage (a difference in points), MRR to 3 decimals."""
    if metric == "hit":
        return f"{value * 100:+.0f}" if signed else f"{value:.0%}"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def span(lo: float, hi: float, metric: str, signed: bool = False) -> str:
    return f"{fmt_score(lo, metric, signed)} to {fmt_score(hi, metric, signed)}"


def ci_tables(cis: dict, retrieval: dict) -> str:
    lines = ["| Mode | Hit@5 | 95% CI | MRR | 95% CI |", "|---|---|---|---|---|"]
    for label in CONFIGS:
        rows = list(retrieval[label].values())
        hit, mrr = cis["modes"][label]["hit"], cis["modes"][label]["mrr"]
        lines.append(
            f"| {label} | {mean(rows, 'hit')[0]:.0%} | {span(*hit, 'hit')} "
            f"| {mean(rows, 'mrr')[0]:.3f} | {span(*mrr, 'mrr')} |"
        )
    lines += ["", "| Difference | Hit@5 (pts) | 95% CI | MRR | 95% CI | Clear? |", "|---|---|---|---|---|---|"]
    for row in cis["differences"]:
        a, b = row["modes"]
        hit, mrr = row["hit"], row["mrr"]
        # "Clear" = the interval excludes 0 for at least one of the two metrics.
        clear = any(r["ci"][0] > 0 or r["ci"][1] < 0 for r in (hit, mrr))
        lines.append(
            f"| {a} vs {b} | {hit['diff'] * 100:+.0f} | {span(*hit['ci'], 'hit', True)} "
            f"| {mrr['diff']:+.3f} | {span(*mrr['ci'], 'mrr', True)} | {'yes' if clear else 'no'} |"
        )
    return "\n".join(lines)


def summary_table(retrieval: dict, testset: list[dict], metrics: list[str], ids=None) -> str:
    ids = ids or [item["id"] for item in testset]
    header = "| Mode | " + " | ".join(METRIC_NAMES[m] for m in metrics) + " |"
    lines = [header, "|" + "---|" * (len(metrics) + 1)]
    for label in CONFIGS:
        rows = [retrieval[label][qid] for qid in ids]
        cells = []
        for metric in metrics:
            value, n = mean(rows, metric)
            as_percent = metric == "hit"
            cell = "n/a" if value is None else (f"{value:.0%}" if as_percent else f"{value:.3f}")
            if value is not None and n < len(rows) and metric != "faithfulness":
                cell += f" (n={n})"
            cells.append(cell)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def latency_line(retrieval: dict) -> str:
    parts = []
    for label in CONFIGS:
        rows = retrieval[label].values()
        parts.append(f"{label} {sum(r['latency'] for r in rows) / len(rows):.2f} s")
    return "Mean retrieval latency per question (CPU): " + ", ".join(parts) + "."


def per_question_table(retrieval: dict, testset: list[dict], with_llm: bool) -> str:
    labels = list(CONFIGS)
    header = "| ID | Style | Question | " + " | ".join(f"Rank: {l}" for l in labels)
    if with_llm:
        header += " | " + " | ".join(f"Accuracy: {l}" for l in LLM_CONFIGS)
    columns = 3 + len(labels) + (len(LLM_CONFIGS) if with_llm else 0)
    lines = [header + " |", "|" + "---|" * columns]
    for item in testset:
        rows = [retrieval[label][item["id"]] for label in labels]
        cells = [str(r["first_rank"]) if r["first_rank"] else "miss" for r in rows]
        if with_llm:
            cells += [fmt(retrieval[label][item["id"]].get("accuracy")) for label in LLM_CONFIGS]
        lines.append(f"| {item['id']} | {item['style']} | {item['question']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_json(path: Path, retrieval: dict, testset: list[dict], metrics: list[str], complete: bool) -> None:
    """The same numbers as the Markdown report, for the app's Evaluation tab."""
    summary = {}
    for label in CONFIGS:
        rows = list(retrieval[label].values())
        summary[label] = {metric: mean(rows, metric)[0] for metric in metrics}
        summary[label]["latency_s"] = sum(r["latency"] for r in rows) / len(rows)
    by_style = {
        style: {
            label: {m: mean([retrieval[label][i["id"]] for i in testset if i["style"] == style], m)[0]
                    for m in ("hit", "mrr")}
            for label in CONFIGS
        }
        for style in ("keyword", "paraphrase")
    }
    per_question = [
        {
            "id": item["id"],
            "style": item["style"],
            "question": item["question"],
            "first_rank": {label: retrieval[label][item["id"]]["first_rank"] for label in CONFIGS},
        }
        for item in testset
    ]
    data = {
        "date": date.today().isoformat(),
        "rerank_model": config.RERANK_MODEL,
        "metrics": metrics,
        "llm_modes": LLM_CONFIGS,
        "complete": complete,
        "summary": summary,
        "confidence_intervals": confidence_intervals(retrieval, testset),
        "by_style": by_style,
        "per_question": per_question,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_report(
    path: Path, retrieval: dict, testset: list[dict], chunk_count: int, with_llm: bool, complete: bool
) -> str:
    metrics = RETRIEVAL_METRICS + (LLM_METRICS if with_llm else [])
    by_style = {}
    for style in ("keyword", "paraphrase"):
        ids = [item["id"] for item in testset if item["style"] == style]
        by_style[style] = (len(ids), summary_table(retrieval, testset, ["hit", "mrr"], ids))
    answered = {
        label: sum(1 for r in retrieval[label].values() if r.get("answer") not in (None, NOT_FOUND))
        for label in LLM_CONFIGS
    }

    parts = [
        "# Evaluation results",
        "",
        f"Run on {date.today().isoformat()} with `python eval/run_eval.py`: "
        f"{len(testset)} questions, {chunk_count} chunks from 32 documents, top {config.TOP_K} chunks per question, "
        f"reranker `{config.RERANK_MODEL}`.",
        "",
        "## Summary",
        "",
        summary_table(retrieval, testset, metrics),
        "",
        latency_line(retrieval),
    ]
    if with_llm:
        parts += [
            "",
            f"Answer metrics are measured for {', '.join(LLM_CONFIGS)} only (the free judge quota allows "
            "one mode). Answered (not \"couldn't find\"): "
            + ", ".join(f"{label} {n}/{len(testset)}" for label, n in answered.items())
            + ". Faithfulness is averaged over answered questions only, since a "
            "\"couldn't find\" reply makes no claims to check.",
        ]
        if not complete:
            parts += ["", "**Incomplete:** a rate limit stopped the LLM run; rerun to finish."]
    else:
        parts += ["", "Retrieval metrics only (`--retrieval-only`); `python eval/run_eval.py` adds the answer metrics."]
    parts += [
        "",
        "## Is the difference real? (95% bootstrap confidence intervals)",
        "",
        f"Resampling the {len(testset)} questions {BOOTSTRAP_SAMPLES:,} times. Differences are paired: "
        "both modes are scored on the same resampled questions. *Clear* means the interval for Hit@5 "
        "or MRR excludes 0.",
        "",
        ci_tables(confidence_intervals(retrieval, testset), retrieval),
    ]
    for style, (n, table) in by_style.items():
        parts += ["", f"## {style.capitalize()} questions ({n})", "", table]
    parts += [
        "",
        "## Per question",
        "",
        "Rank = position of the first relevant chunk in the top 5 (miss = none retrieved).",
        "",
        per_question_table(retrieval, testset, with_llm),
        "",
        "## Method",
        "",
        "- **Test set** (`eval/testset.jsonl`): questions written from the corpus, each with a "
        "reference answer and the exact phrase(s) from the source document that answer it. "
        "*Keyword* questions use the part code or parameter name; *paraphrase* questions describe "
        "it in other words.",
        "- **Relevant chunks:** every chunk of the source document whose section title + text "
        "contains an evidence phrase.",
        "- **Hit@5:** share of questions with at least one relevant chunk in the top 5. "
        "**MRR:** mean of 1 / rank of the first relevant chunk (0 when missed).",
        "- **Context recall / precision:** ragas `IDBasedContextRecall` (share of relevant chunks "
        "retrieved) and `IDBasedContextPrecision` (share of retrieved chunks that are relevant). "
        "Precision is capped well below 1, because most questions have only one or two relevant "
        "chunks among the five retrieved.",
    ]
    if with_llm:
        parts += [
            f"- **Answers** generated by `{config.GROQ_MODEL}` (Groq) from each mode's top 5 chunks; "
            f"judged by `{config.JUDGE_MODEL}` (Groq), a different model, so it isn't grading its own answers.",
            "- **Faithfulness:** ragas `Faithfulness` (share of the answer's claims supported by the "
            "retrieved chunks). **Answer accuracy:** ragas `AnswerAccuracy` (two judge prompts rate the "
            "answer against the reference as 0, 0.5 or 1, averaged).",
        ]
    report = "\n".join(parts) + "\n"
    path.write_text(report, encoding="utf-8")
    write_json(path.with_suffix(".json"), retrieval, testset, metrics, complete)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare dense, hybrid and hybrid+rerank retrieval.")
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM answer metrics")
    parser.add_argument(
        "--output", type=Path, default=RESULTS_PATH, help="report path (a .json copy is written next to it)"
    )
    args = parser.parse_args()

    testset = load_jsonl(TESTSET_PATH)
    chunks = load_jsonl(CHUNKS_PATH)
    retrieval = run_retrieval(testset, chunks)
    # Retrieval is done; free the store so the CLI or app can use it while the
    # (slow, rate-limited) LLM part runs.
    close_client()

    complete = True
    if not args.retrieval_only:
        cache = load_cache()
        complete = asyncio.run(run_answers(testset, retrieval, cache))

    metrics = RETRIEVAL_METRICS + ([] if args.retrieval_only else LLM_METRICS)
    write_report(args.output, retrieval, testset, len(chunks), not args.retrieval_only, complete)
    print("\n" + summary_table(retrieval, testset, metrics))
    print(latency_line(retrieval))
    print(f"\nFull report: {args.output}")


if __name__ == "__main__":
    main()
