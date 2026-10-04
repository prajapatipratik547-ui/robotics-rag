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

# Models (all run locally on CPU)
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"
RERANK_MODEL = "BAAI/bge-reranker-base"

# Secrets (read from .env, never hardcoded)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
