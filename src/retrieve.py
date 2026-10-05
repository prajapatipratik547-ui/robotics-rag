"""Retrieval: find the chunks most relevant to a question.

Three modes, selectable so they can be compared:
  dense   semantic similarity of bge-small embeddings (captures meaning)
  sparse  BM25 keyword match (captures exact tokens like "A1M8", "0x68")
  hybrid  both, fused with Reciprocal Rank Fusion (RRF)

Any mode can be followed by cross-encoder reranking (on by default).

Compare the modes side by side:
  python src/retrieve.py "What baud rate does the A1M8 use?"
"""

import argparse
import atexit
import sys
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models

import config
from src.ingest import embedding_input


@dataclass
class Hit:
    """One retrieved chunk plus its retrieval score (and rerank score, if
    reranked: a raw relevance logit where higher is better and >0 roughly
    means 'relevant')."""

    chunk_id: str
    source_file: str
    section: str
    page: int | None
    text: str
    score: float
    rerank_score: float | None = None

    @property
    def citation(self) -> str:
        """Human-readable source label, e.g. 'mpu6050_datasheet.pdf, page 12'."""
        return f"{self.source_file}, {self.section}"


# Loaded once per process: model loading and opening the local Qdrant store
# are slow, and local mode allows only one client per storage folder.
@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    client = QdrantClient(path=str(config.QDRANT_PATH))
    # Close explicitly at exit; otherwise the client is garbage-collected
    # during interpreter shutdown and prints a spurious ImportError.
    atexit.register(client.close)
    return client


def close_client() -> None:
    """Release the local Qdrant store so another process can open it
    (local mode allows only one process at a time)."""
    if get_client.cache_info().currsize:
        get_client().close()
        get_client.cache_clear()


@lru_cache(maxsize=1)
def get_dense_model() -> TextEmbedding:
    return TextEmbedding(config.DENSE_MODEL, cache_dir=str(config.MODEL_CACHE_DIR))


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(config.SPARSE_MODEL, cache_dir=str(config.MODEL_CACHE_DIR))


@lru_cache(maxsize=1)
def get_reranker() -> TextCrossEncoder:
    return TextCrossEncoder(config.RERANK_MODEL, cache_dir=str(config.MODEL_CACHE_DIR))


def _dense_query(query: str) -> list[float]:
    return next(get_dense_model().query_embed(query)).tolist()


def _sparse_query(query: str) -> models.SparseVector:
    # query_embed gives each query term weight 1; Qdrant multiplies in the
    # IDF (rarer term -> bigger weight) because the collection's sparse
    # vector was created with Modifier.IDF.
    embedding = next(get_sparse_model().query_embed(query))
    return models.SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())


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
    response = get_client().query_points(
        config.COLLECTION_NAME,
        query=_dense_query(query),
        using=config.DENSE_VECTOR_NAME,
        limit=k,
        with_payload=True,
    )
    return _to_hits(response.points)


def sparse_search(query: str, k: int = config.TOP_K) -> list[Hit]:
    """BM25: score chunks by the question words they contain, weighting rare
    words (part codes, register names) far above common ones."""
    response = get_client().query_points(
        config.COLLECTION_NAME,
        query=_sparse_query(query),
        using=config.SPARSE_VECTOR_NAME,
        limit=k,
        with_payload=True,
    )
    return _to_hits(response.points)


def hybrid_search(query: str, k: int = config.TOP_K, prefetch_k: int = config.HYBRID_PREFETCH_K) -> list[Hit]:
    """Run dense and sparse search inside Qdrant (the two prefetches), then
    fuse the two ranked lists with Reciprocal Rank Fusion.

    RRF ignores the raw scores (cosine and BM25 are on incomparable scales)
    and uses only each chunk's rank in each list: a chunk earns 1/(k + rank - 1)
    from every list it appears in (Qdrant's default k = 2), and the sums decide
    the final order. A chunk near the top of both lists beats one that
    tops just one list."""
    response = get_client().query_points(
        config.COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=_dense_query(query), using=config.DENSE_VECTOR_NAME, limit=prefetch_k),
            models.Prefetch(query=_sparse_query(query), using=config.SPARSE_VECTOR_NAME, limit=prefetch_k),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
        with_payload=True,
    )
    return _to_hits(response.points)


SEARCH_FUNCTIONS = {"dense": dense_search, "sparse": sparse_search, "hybrid": hybrid_search}


def search(query: str, k: int = config.TOP_K, mode: str = config.DEFAULT_MODE) -> list[Hit]:
    if mode not in SEARCH_FUNCTIONS:
        raise ValueError(f"Unknown retrieval mode {mode!r}; choose from {config.RETRIEVAL_MODES}")
    return SEARCH_FUNCTIONS[mode](query, k)


def rerank(query: str, hits: list[Hit], k: int = config.TOP_K) -> list[Hit]:
    """Reorder candidates with a cross-encoder and keep the best k.

    Dense search compares two vectors made separately (question alone, chunk
    alone). A cross-encoder instead reads the question and the chunk together
    in one pass, so it can check whether this chunk actually answers this
    question. Far more accurate, but too slow to run over the whole corpus,
    hence retrieve-then-rerank: cheap search narrows to ~20, this picks 5.

    It scores the same header + text that was embedded, so it knows which
    part a datasheet page belongs to."""
    if not hits:
        return []
    documents = [embedding_input(asdict(hit)) for hit in hits]
    scores = list(get_reranker().rerank(query, documents))
    reranked = [replace(hit, rerank_score=float(s)) for hit, s in zip(hits, scores)]
    reranked.sort(key=lambda hit: hit.rerank_score, reverse=True)
    return reranked[:k]


def retrieve(
    query: str,
    k: int = config.TOP_K,
    mode: str = config.DEFAULT_MODE,
    use_rerank: bool = config.DEFAULT_RERANK,
) -> list[Hit]:
    """The retrieval entry point used by the pipeline: search in the given
    mode, optionally widening to RERANK_CANDIDATES and reranking down to k."""
    if not use_rerank:
        return search(query, k, mode)
    candidates = search(query, max(k, config.RERANK_CANDIDATES), mode)
    return rerank(query, candidates, k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare retrieval modes for one query.")
    parser.add_argument("query", nargs="+")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    args = parser.parse_args()
    query = " ".join(args.query)

    def show(label: str, hits: list[Hit]) -> None:
        print(f"\n{label}:")
        for rank, hit in enumerate(hits, start=1):
            preview = " ".join(hit.text.split())[:80]
            rerank_col = f"  rerank={hit.rerank_score:6.2f}" if hit.rerank_score is not None else ""
            print(f"  {rank}. {hit.score:7.3f}{rerank_col}  {hit.chunk_id:<38} {preview}")

    print(f"Query: {query}")
    for mode in config.RETRIEVAL_MODES:
        show(mode, search(query, args.k, mode))
    show("hybrid + rerank", retrieve(query, args.k, "hybrid", use_rerank=True))


if __name__ == "__main__":
    main()
