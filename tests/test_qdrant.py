"""Stage 0 smoke test: prove Qdrant runs locally (no Docker).

Creates a local-mode collection, inserts 3 text points, runs a
similarity search, and prints the results.

Run:  python tests/test_qdrant.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

import config

COLLECTION = "smoke_test"

TEXTS = [
    "The RPLiDAR A1M8 is a 360-degree 2D laser scanner for robot mapping.",
    "The MPU6050 combines a 3-axis gyroscope and a 3-axis accelerometer.",
    "The L298N is a dual H-bridge driver for controlling DC motors.",
]
QUERY = "Which sensor measures angular velocity and acceleration?"
EXPECTED = TEXTS[1]


def main() -> None:
    embedder = TextEmbedding(config.DENSE_MODEL, cache_dir=str(config.MODEL_CACHE_DIR))
    vectors = list(embedder.embed(TEXTS))
    dim = len(vectors[0])

    client = QdrantClient(path=str(config.QDRANT_PATH))
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )
    client.upsert(
        COLLECTION,
        points=[
            models.PointStruct(id=i, vector=vec.tolist(), payload={"text": text})
            for i, (text, vec) in enumerate(zip(TEXTS, vectors))
        ],
    )

    query_vec = next(embedder.query_embed(QUERY)).tolist()
    hits = client.query_points(COLLECTION, query=query_vec, limit=3).points

    print(f"Query: {QUERY}\n")
    for rank, hit in enumerate(hits, start=1):
        print(f"{rank}. score={hit.score:.4f}  {hit.payload['text']}")

    client.delete_collection(COLLECTION)
    client.close()

    top = hits[0].payload["text"]
    if top != EXPECTED:
        print(f"\nFAIL: expected top match to be:\n  {EXPECTED}")
        sys.exit(1)
    print("\nPASS: top match is the expected text.")


if __name__ == "__main__":
    main()
