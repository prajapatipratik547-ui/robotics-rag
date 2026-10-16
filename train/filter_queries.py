"""Step 2 of fine-tuning: keep only the synthetic queries worth training on.

Two filters, applied to train/data/queries.jsonl:

1. Leakage: drop any query that shares half or more of its words with a test
   question (token Jaccard >= 0.5) or means nearly the same (embedding cosine
   >= 0.85, checked last over the whole file), and any query generated for
   more than one chunk (too generic to point at one passage).
2. Round-trip consistency: search the corpus with the query (hybrid + the
   cross-encoder reranker, on a separate experiment index) and keep it only if
   its own chunk comes back in the top 3. Vague queries such as "rplidar a1m8
   datasheet" match many chunks and drop out.

Runs in the main environment. Resumable.

    QDRANT_PATH=qdrant_exp/base RERANK_MODEL=jinaai/jina-reranker-v1-tiny-en \
        python train/filter_queries.py [--queries train/data/queries_llm.jsonl --pairs train/data/pairs_llm.jsonl]
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.retrieve import get_dense_model, retrieve
from src.testset import TESTSET_PATH, load_jsonl

QUERIES_PATH = ROOT / "train" / "data" / "queries.jsonl"
PAIRS_PATH = ROOT / "train" / "data" / "pairs.jsonl"
MAX_TEST_OVERLAP = 0.5
MAX_TEST_SIMILARITY = 0.85  # strict: same-fact rewordings of test questions score ~0.85-0.96
KEEP_TOP = 3


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    parser.add_argument("--pairs", type=Path, default=PAIRS_PATH, help="output")
    args = parser.parse_args()
    if config.QDRANT_PATH.name == "qdrant_data":
        sys.exit("Set QDRANT_PATH to an experiment index (e.g. qdrant_exp/base), not the app's index.")
    rows = [r for r in load_jsonl(args.queries) if r["query"]]  # None: the generator had no question
    test_words = [words(item["question"]) for item in load_jsonl(TESTSET_PATH)]
    chunks_per_query = Counter(row["query"].lower() for row in rows)
    done = {(r["chunk_id"], r["query"]) for r in load_jsonl(args.pairs)} if args.pairs.exists() else set()
    print(f"{len(rows)} queries, {len(done)} already filtered; reranker {config.RERANK_MODEL}")

    start, checked = time.time(), 0
    with args.pairs.open("a", encoding="utf-8") as out:
        for row in rows:
            if (row["chunk_id"], row["query"]) in done:
                continue
            query_words = words(row["query"])
            if any(jaccard(query_words, t) >= MAX_TEST_OVERLAP for t in test_words):
                verdict = "test_overlap"
            elif chunks_per_query[row["query"].lower()] > 1:
                verdict = "generic"
            else:
                top = [hit.chunk_id for hit in retrieve(row["query"], KEEP_TOP, "hybrid", True)]
                verdict = "kept" if row["chunk_id"] in top else "not_found"
                checked += 1
            out.write(json.dumps({**row, "verdict": verdict}, ensure_ascii=False) + "\n")
            if checked and checked % 100 == 0 and verdict in ("kept", "not_found"):
                out.flush()
                print(f"{checked} searched, {checked / (time.time() - start):.1f}/s", flush=True)

    drop_test_paraphrases(args.pairs, load_jsonl(TESTSET_PATH))
    verdicts = Counter(r["verdict"] for r in load_jsonl(args.pairs))
    print("verdicts:", dict(verdicts))


def drop_test_paraphrases(pairs_path: Path, testset: list[dict]) -> None:
    """Relabel kept queries that mean the same as a test question in other
    words ("selectable full-scale ranges for the MPU-60X0 gyroscope" vs "What
    full-scale ranges can the gyroscope on the MPU-6050 be set to?"), which
    word overlap misses. Cosine >= MAX_TEST_SIMILARITY under the base model.
    Idempotent: safe to run on an already-filtered file."""
    import numpy as np

    rows = load_jsonl(pairs_path)
    kept = [r for r in rows if r["verdict"] == "kept"]
    if not kept:
        return
    model = get_dense_model()
    tests = np.array(list(model.query_embed([item["question"] for item in testset])))
    queries = np.array(list(model.query_embed([r["query"] for r in kept])))
    for row, similarity in zip(kept, (queries @ tests.T).max(axis=1)):
        if similarity >= MAX_TEST_SIMILARITY:
            row["verdict"] = "test_similar"
    pairs_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
