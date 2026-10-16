"""Step 1 of fine-tuning: synthetic training queries, no human labels.

A local doc2query model (T5 trained on MS MARCO) reads each chunk, formatted
exactly as it is embedded, and samples a few questions the chunk answers.
Runs on CPU in the training environment (requirements-train.txt), ~1 s per
query. Resumable: chunks already in the output file are skipped.

    python train/make_queries.py            # all chunks, 5 queries each
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.passages import embedding_input

CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
QUERIES_PATH = ROOT / "train" / "data" / "queries.jsonl"
MODEL = "doc2query/msmarco-t5-base-v1"
BATCH_CHUNKS = 8


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-chunk", type=int, default=5, help="queries sampled per chunk")
    args = parser.parse_args()

    chunks = load_jsonl(CHUNKS_PATH)
    QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = {row["chunk_id"] for row in load_jsonl(QUERIES_PATH)} if QUERIES_PATH.exists() else set()
    todo = [c for c in chunks if c["chunk_id"] not in done]
    print(f"{len(chunks)} chunks, {len(done)} done, {len(todo)} to go")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).eval()
    torch.manual_seed(0)  # sampling, so fix the seed for a reproducible set
    start = time.time()
    with QUERIES_PATH.open("a", encoding="utf-8") as out, torch.no_grad():
        for i in range(0, len(todo), BATCH_CHUNKS):
            batch = todo[i : i + BATCH_CHUNKS]
            encoded = tokenizer(
                [embedding_input(c) for c in batch], max_length=384, truncation=True, padding=True, return_tensors="pt"
            )
            generated = model.generate(
                **encoded, max_length=64, do_sample=True, top_p=0.95, top_k=10, num_return_sequences=args.per_chunk
            )
            queries = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for j, chunk in enumerate(batch):
                own = queries[j * args.per_chunk : (j + 1) * args.per_chunk]
                for query in dict.fromkeys(q.strip() for q in own if q.strip()):  # dedupe, keep order
                    out.write(json.dumps({"chunk_id": chunk["chunk_id"], "query": query}, ensure_ascii=False) + "\n")
            out.flush()
            n = i + len(batch)
            rate = n / (time.time() - start)
            print(f"{n}/{len(todo)} chunks, {rate:.2f} chunks/s, ~{(len(todo) - n) / rate / 60:.0f} min left", flush=True)


if __name__ == "__main__":
    main()
