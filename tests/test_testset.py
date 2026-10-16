"""Evidence-based labelling of answer-bearing chunks, and the test set's integrity."""

import pytest

from src.testset import TESTSET_PATH, load_jsonl, normalize, relevant_chunk_ids

CHUNKS = [
    {"chunk_id": "a.pdf#0000", "source_file": "a.pdf", "section": "page 1", "text": "The I2C address is\n1101001  when AD0 is high."},
    {"chunk_id": "a.pdf#0001", "source_file": "a.pdf", "section": "page 2", "text": "Unrelated text."},
    {"chunk_id": "b.html#0000", "source_file": "b.html", "section": "update_min_d", "text": "Type: double Default: 0.25"},
]


def test_normalize_collapses_whitespace_and_case():
    assert normalize("The  I2C\nAddress") == "the i2c address"


def test_evidence_matches_across_line_breaks_and_case():
    item = {"id": "q", "evidence": [{"source_file": "a.pdf", "text": "address is 1101001 when"}]}
    assert relevant_chunk_ids(item, CHUNKS) == ["a.pdf#0000"]


def test_evidence_can_match_the_section_heading():
    item = {"id": "q", "evidence": [{"source_file": "b.html", "text": "update_min_d type: double"}]}
    assert relevant_chunk_ids(item, CHUNKS) == ["b.html#0000"]


def test_evidence_only_matches_its_own_document():
    item = {"id": "q", "evidence": [{"source_file": "a.pdf", "text": "default: 0.25"}]}
    with pytest.raises(ValueError):
        relevant_chunk_ids(item, CHUNKS)


def test_testset_is_well_formed():
    items = load_jsonl(TESTSET_PATH)
    assert len(items) == 85
    assert len({i["id"] for i in items}) == len(items)
    for item in items:
        assert item["style"] in {"keyword", "paraphrase"}
        assert item["question"].strip() and item["reference"].strip() and item["evidence"]
