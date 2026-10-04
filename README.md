# Robotics Hybrid RAG

A question-answering system for robotics hardware and software that answers
from real documents (datasheets, ROS2/Nav2 docs) with citations, using hybrid
dense + BM25 retrieval, RRF fusion, and cross-encoder reranking.

> **Current stage: 3, Hybrid retrieval (complete).** Retrieval mode is
> selectable: `dense` (the Stage 2 baseline), `sparse` (BM25), or `hybrid`
> (both, fused with Reciprocal Rank Fusion; the default). Answers come from
> the top-5 chunks via Gemini 2.5 Flash, with citations.
> The full README arrives in Stage 6. The build plan is in `build.rag.md`.
>
> `data/raw/` currently holds three **placeholder** files (`*_PLACEHOLDER.*`)
> for testing only. They will be replaced with real datasheets and docs.

## Quick start

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then set GEMINI_API_KEY (free key: aistudio.google.com)
python tests/test_qdrant.py # Stage 0 smoke test
python src/ingest.py        # Stage 1: load, chunk, embed, store data/raw/
python src/rag.py "What is the scan range of the RPLiDAR A1M8?"
python src/rag.py           # interactive mode; add --show-context to see retrieved chunks
python src/rag.py --mode dense "Is MPPI supported?"   # compare: dense | sparse | hybrid
python src/retrieve.py "Is MPPI supported?"           # side-by-side ranking of all 3 modes
```

Name source files descriptively (e.g. `rplidar_a1m8_datasheet.pdf`): the file
name becomes the document title that is prepended to every chunk before
embedding.
