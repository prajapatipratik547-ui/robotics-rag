# Robotics Hybrid RAG

A question-answering system for robotics hardware and software that answers
from real documents (datasheets, ROS2/Nav2 docs) with citations, using hybrid
dense + BM25 retrieval, RRF fusion, and cross-encoder reranking.

> **Current stage: 1, Ingestion (complete).** Documents in `data/raw/` are
> chunked and stored in local Qdrant with dense + BM25 vectors.
> The full README arrives in Stage 6. The build plan is in `build.rag.md`.
>
> `data/raw/` currently holds three **placeholder** files (`*_PLACEHOLDER.*`)
> for testing only. They will be replaced with real datasheets and docs.

## Quick start

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then add your LLM key (needed from Stage 2)
python tests/test_qdrant.py # Stage 0 smoke test
python src/ingest.py        # Stage 1: load, chunk, embed, store data/raw/
```

Name source files descriptively (e.g. `rplidar_a1m8_datasheet.pdf`): the file
name becomes the document title that is prepended to every chunk before
embedding.
