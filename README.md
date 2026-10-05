# Robotics Hybrid RAG

A question-answering system for robotics hardware and software that answers
from real documents (datasheets, ROS2/Nav2 docs) with citations, using hybrid
dense + BM25 retrieval, RRF fusion, and cross-encoder reranking.

> **Current stage: 5, Evaluation (complete).** Retrieval mode is selectable:
> `dense` (the Stage 2 baseline), `sparse` (BM25), or `hybrid` (both, fused
> with Reciprocal Rank Fusion; the default). By default the top 20 candidates
> are then reranked by the `bge-reranker-base` cross-encoder and the best 5
> go to the LLM (Groq `qwen/qwen3.8-27b`), which answers with citations.
> `python eval/run_eval.py` compares dense, hybrid and hybrid+rerank on a
> 26-question test set; results are in [eval/results.md](eval/results.md).
> Headline: hybrid retrieval raised Hit@5 from 88% to 96% and MRR from 0.815
> to 0.891 over dense-only; reranking raised MRR further to 0.905. The
> LLM-judged answer metrics are still being filled in, as the judge model's
> free quota (200K tokens/day) allows.
> The full README arrives in Stage 6. The build plan is in `build.rag.md`.

## Corpus

32 documents, 906 chunks:

- **9 datasheets:** RPLiDAR A1M8, Raspberry Pi 4B, Arduino Mega 2560, MPU6050
  (product spec + register map), NEO-6M GPS, L298N, HC-SR04, MG996R.
- **23 Nav2 documentation pages** (ROS 2 Jazzy): navigation concepts, robot
  setup guides, the core servers, costmaps, AMCL, controllers (RPP, MPPI),
  planners (NavFn, Smac Hybrid), tuning, and the SLAM and GPS tutorials.

The documents are copyrighted by their publishers, so they are not in git.
`python data/download_corpus.py` downloads them into `data/raw/` and lists
the source of every file.

## Quick start

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then set GROQ_API_KEY (free key: console.groq.com)
python tests/test_qdrant.py # Stage 0 smoke test
python data/download_corpus.py  # fetch the 32 source documents into data/raw/
python src/ingest.py        # Stage 1: load, chunk, embed, store data/raw/
python src/rag.py "What is the scan range of the RPLiDAR A1M8?"
python src/rag.py           # interactive mode; add --show-context to see retrieved chunks
python src/rag.py --mode dense "Is MPPI supported?"   # compare: dense | sparse | hybrid
python src/rag.py --no-rerank "A1M8 scan frequency"   # turn reranking off
python src/retrieve.py "A1M8 scan frequency"          # side-by-side: 3 modes + hybrid+rerank
python eval/run_eval.py --retrieval-only  # Stage 5 retrieval metrics, no LLM calls (~2 min)
python eval/run_eval.py                   # plus answer metrics judged by a Groq LLM (slow: rate limits)
```

The first run downloads the models (~1.2 GB, mostly the reranker) into
`.fastembed_cache/`.

The LLM is Groq by default (free tier on this key: 1,000 requests/day and
8,000 tokens/minute per model). Gemini also works: set `LLM_PROVIDER =
"gemini"` in `config.py` and `GEMINI_API_KEY` in `.env`, but its free tier
allowed only 20 requests/day here. The eval caches LLM answers and judge
scores in `eval/cache/`, so a run stopped by a rate limit resumes on rerun.

Name source files descriptively (e.g. `rplidar_a1m8_datasheet.pdf`): the file
name becomes the document title that is prepended to every chunk before
embedding.
