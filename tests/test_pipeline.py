"""End to end on a toy corpus: embed, store in a temporary local Qdrant, and
retrieve in every mode, reranked too. Uses the real models (downloaded once,
~0.3 GB) but never touches the app's index."""

import pytest
from qdrant_client import QdrantClient

import config
import src.retrieve as retrieve_module
from src.ingest import embed_and_store

CHUNKS = [
    {"chunk_id": "mpu6050.pdf#0000", "source_file": "mpu6050.pdf", "section": "page 15", "page": 15,
     "text": "The slave address of the MPU-60X0 is b110100X. The LSB is set by pin AD0: b1101000 when AD0 is "
             "logic low and b1101001 when AD0 is logic high."},
    {"chunk_id": "rplidar.pdf#0000", "source_file": "rplidar.pdf", "section": "page 8", "page": 8,
     "text": "Distance range 0.15-12 m on white objects. Sample frequency 8000 Hz, scan rate 5.5 Hz typical."},
    {"chunk_id": "nav2_amcl.html#0000", "source_file": "nav2_amcl.html", "section": "update_min_d", "page": None,
     "text": "Type: double Default: 0.25 Translational movement required before performing a filter update."},
    {"chunk_id": "l298.pdf#0000", "source_file": "l298.pdf", "section": "page 3", "page": 3,
     "text": "Absolute maximum ratings: power supply 50 V, logic supply voltage 7 V, DC output current 2 A per channel."},
]


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    client = QdrantClient(path=str(tmp_path_factory.mktemp("qdrant")))
    embed_and_store(client, CHUNKS)
    yield client
    client.close()


@pytest.fixture(autouse=True)
def use_store(store, monkeypatch):
    monkeypatch.setattr(retrieve_module, "get_client", lambda: store)
    # A small reranker keeps the test fast; the pipeline is the same.
    monkeypatch.setattr(config, "RERANK_MODEL", "jinaai/jina-reranker-v1-tiny-en")
    retrieve_module.get_reranker.cache_clear()


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_each_mode_finds_the_answer_first(mode):
    hits = retrieve_module.retrieve("What is the I2C address of the MPU-6050 when AD0 is high?", 3, mode, False)
    assert hits[0].chunk_id == "mpu6050.pdf#0000"


def test_keyword_search_finds_an_exact_parameter_name():
    hits = retrieve_module.retrieve("update_min_d", 1, "sparse", False)
    assert hits[0].chunk_id == "nav2_amcl.html#0000"


def test_reranking_scores_and_orders_the_hits():
    hits = retrieve_module.retrieve("How far does the RPLiDAR measure?", 3, "hybrid", True)
    assert hits[0].chunk_id == "rplidar.pdf#0000"
    scores = [h.rerank_score for h in hits]
    assert all(s is not None for s in scores) and scores == sorted(scores, reverse=True)
