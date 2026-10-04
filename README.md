# Robotics Hybrid RAG

A question-answering system for robotics hardware and software that answers
from real documents (datasheets, ROS2/Nav2 docs) with citations, using hybrid
dense + BM25 retrieval, RRF fusion, and cross-encoder reranking.

> **Current stage: 0, Setup (complete).** Local Qdrant is verified working.
> The full README arrives in Stage 6. The build plan is in `build.rag.md`.

## Quick start

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then add your LLM key (needed from Stage 2)
python tests/test_qdrant.py # Stage 0 smoke test
```
