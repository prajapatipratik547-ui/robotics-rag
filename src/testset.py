"""The evaluation test set, and which chunks answer each of its questions.

Shared by eval/run_eval.py and the app's Compare tab (which marks the
answer-bearing chunks for test-set questions), so both use the same rule.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

TESTSET_PATH = config.ROOT_DIR / "eval" / "testset.jsonl"
CHUNKS_PATH = config.PROCESSED_DATA_DIR / "chunks.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def relevant_chunk_ids(item: dict, chunks: list[dict]) -> list[str]:
    """Every chunk whose section title + text contains one of the item's
    evidence phrases, within that phrase's source file. Matching on text
    rather than storing chunk IDs keeps the test set valid if chunking changes."""
    ids: list[str] = []
    for evidence in item["evidence"]:
        phrase = normalize(evidence["text"])
        for chunk in chunks:
            if chunk["source_file"] != evidence["source_file"] or chunk["chunk_id"] in ids:
                continue
            if phrase in normalize(f"{chunk['section']} {chunk['text']}"):
                ids.append(chunk["chunk_id"])
    if not ids:
        raise ValueError(f"{item['id']}: no chunk contains its evidence; check {TESTSET_PATH.name}")
    return ids
