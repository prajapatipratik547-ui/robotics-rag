"""Central settings: model names, paths, and constants.

Every other module imports from here, so changing a model or path
only ever happens in one place.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
ROOT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
QDRANT_PATH = ROOT_DIR / "qdrant_data"  # local-mode storage, no Docker
MODEL_CACHE_DIR = ROOT_DIR / ".fastembed_cache"  # downloaded models live here, not in %TEMP%

# Qdrant
COLLECTION_NAME = "robotics_docs"
DENSE_VECTOR_NAME = "dense"  # named vector for semantic search
SPARSE_VECTOR_NAME = "bm25"  # named vector for keyword search

# Chunking (counted in the dense model's own tokens).
# bge-small and bge-reranker-base both truncate input at 512 tokens, so a
# 400-token chunk leaves room for the title/section header we prepend and,
# at rerank time, for the question that gets paired with the chunk.
CHUNK_MAX_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 100
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".html", ".htm"}

# Retrieval
TOP_K = 5  # chunks handed to the LLM
RETRIEVAL_MODES = ("dense", "sparse", "hybrid")
DEFAULT_MODE = "hybrid"
# Hybrid: how many candidates each of dense and sparse contributes to RRF
# fusion. Wider than TOP_K so a chunk ranked, say, 8th by one method can
# still win if the other method also ranks it well.
HYBRID_PREFETCH_K = 20
# Reranking: fetch this many candidates, let the cross-encoder reorder them,
# keep the best TOP_K. Bigger = better recall but slower (the cross-encoder
# reads every candidate in full, ~0.1-0.2 s each on CPU).
RERANK_CANDIDATES = 20
DEFAULT_RERANK = True

# LLM (answer generation only): "groq" or "gemini".
# Groq is the default: its free tier on this project's key allows 1,000
# requests/day per model (read from Groq's rate-limit response headers), while
# Gemini's allowed only 20/day, too few to run the Stage 5 eval.
LLM_PROVIDER = "groq"
# qwen3.8-27b with reasoning off answered in ~0.3 s in testing and uses the
# [n] citation format we parse; the gpt-oss models write their own 【n†Lx】
# markers instead.
GROQ_MODEL = "qwen/qwen3.8-27b"
# gemini-2.5-flash with thinking off answered in ~1.2 s versus 8-20 s for the
# 3.x Flash models; grounded answering is reading, not reasoning.
GEMINI_MODEL = "gemini-2.5-flash"

# Evaluation judge (Stage 5): a different model from the generator, so the
# judge isn't grading its own answers. Groq limits are per model, so the
# judge also gets its own quota.
JUDGE_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"  # OpenAI-compatible endpoint, used by ragas

# Models (all run locally on CPU)
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"
# Chunks embedded per batch at ingestion. FastEmbed's default of 256 peaks at
# ~7 GB of RAM on 400-token chunks, which gets the app killed on free hosting.
EMBED_BATCH_SIZE = 16
# The reranker can be swapped with the RERANK_MODEL environment variable: the
# 1 GB bge-reranker-base needs ~2 GB of RAM in the app, more than free hosting allows.
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

# Public demo safety: the most LLM answers the app gives per day across all
# visitors, so a public link can't use up the Groq free quota. 0 = no limit
# (local use); set DAILY_LLM_LIMIT in the host's secrets when deploying.
DAILY_LLM_LIMIT = int(os.getenv("DAILY_LLM_LIMIT", "0"))

# Secrets (read from .env, never hardcoded)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
