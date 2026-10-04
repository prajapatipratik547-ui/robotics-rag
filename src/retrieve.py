"""Retrieval: find the chunks most relevant to a question.

Stage 2 has dense (semantic) search only. Sparse, hybrid, and reranking
are added in later stages so each mode can be compared against this one.
"""

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastembed import TextEmbedding
from qdrant_client import QdrantClient

import config


@dataclass
class Hit:
    """One retrieved chunk plus its retrieval score."""

    chunk_id: str
    source_file: str
    section: str
    page: int | None
    text: str
    score: float

    @property
    def citation(self) -> str:
        """Human-readable source label, e.g. 'mpu6050_datasheet.pdf, page 12'."""
        return f"{self.source_file}, {self.section}"


# Loaded once per process: model loading and opening the local Qdrant store
# are slow, and local mode allows only one client per storage folder.
@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    return QdrantClient(path=str(config.QDRANT_PATH))


@lru_cache(maxsize=1)
def get_dense_model() -> TextEmbedding:
    return TextEmbedding(config.DENSE_MODEL, cache_dir=str(config.MODEL_CACHE_DIR))


def _to_hits(points) -> list[Hit]:
    return [
        Hit(
            chunk_id=p.payload["chunk_id"],
            source_file=p.payload["source_file"],
            section=p.payload["section"],
            page=p.payload.get("page"),
            text=p.payload["text"],
            score=p.score,
        )
        for p in points
    ]


def dense_search(query: str, k: int = config.TOP_K) -> list[Hit]:
    """Embed the question with the same model used at ingestion, then return
    the k chunks whose dense vectors are closest by cosine similarity."""
    vector = next(get_dense_model().query_embed(query)).tolist()
    response = get_client().query_points(
        config.COLLECTION_NAME,
        query=vector,
        using=config.DENSE_VECTOR_NAME,
        limit=k,
        with_payload=True,
    )
    return _to_hits(response.points)
