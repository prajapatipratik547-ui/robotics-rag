"""Two-fold split of the corpus by document, for the held-out test.

A model trained with fold A's documents left out is scored only on the test
questions about fold A, so it never saw (synthetic queries for) the passages
it is tested on. The split is deterministic and balanced by test questions.
Dependency-free: used by both the training and the main environment.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTSET_PATH = ROOT / "eval" / "testset.jsonl"
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
FOLDS = ("A", "B")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def question_docs(item: dict) -> set[str]:
    return {e["source_file"] for e in item["evidence"]}


def document_folds() -> dict[str, str]:
    """source_file -> "A" or "B". Documents with the most test questions are
    placed first, each into the fold with fewer questions so far (ties: A)."""
    docs = sorted({c["source_file"] for c in load_jsonl(CHUNKS_PATH)})
    count = {d: 0 for d in docs}
    for item in load_jsonl(TESTSET_PATH):
        for d in question_docs(item):
            count[d] += 1
    folds: dict[str, str] = {}
    totals = {f: 0 for f in FOLDS}
    for doc in sorted(docs, key=lambda d: (-count[d], d)):
        fold = min(FOLDS, key=lambda f: (totals[f], f))
        folds[doc] = fold
        totals[fold] += count[doc]
    return folds


def fold_question_ids(fold: str) -> list[str]:
    """Test questions whose evidence lies entirely in the fold's documents."""
    folds = document_folds()
    return [
        item["id"] for item in load_jsonl(TESTSET_PATH) if all(folds[d] == fold for d in question_docs(item))
    ]


if __name__ == "__main__":
    folds = document_folds()
    for fold in FOLDS:
        docs = [d for d, f in folds.items() if f == fold]
        print(f"fold {fold}: {len(docs)} documents, {len(fold_question_ids(fold))} test questions")
