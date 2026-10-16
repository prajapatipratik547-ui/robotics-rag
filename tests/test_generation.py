"""Passage formatting and the citation handling around the LLM (no LLM calls)."""

from src.generate import build_prompt, drop_invalid_citations, extract_citations
from src.passages import doc_title, embedding_input
from src.retrieve import Hit


def hit(n: int) -> Hit:
    return Hit(f"doc.pdf#{n:04d}", "doc.pdf", f"page {n}", n, f"text {n}", score=1.0)


def test_doc_title_from_file_name():
    assert doc_title("rplidar_a1m8_datasheet.pdf") == "rplidar a1m8 datasheet"
    assert doc_title("nav2_costmap-2d.html") == "nav2 costmap 2d"


def test_embedding_input_prefixes_title_and_section():
    chunk = {"source_file": "nav2_amcl.html", "section": "update_min_d", "text": "Default: 0.25"}
    assert embedding_input(chunk) == "nav2 amcl | update_min_d\n\nDefault: 0.25"


def test_prompt_numbers_passages_with_their_source():
    prompt = build_prompt("Q?", [hit(1), hit(2)])
    assert "[1] (source: doc.pdf, page 1)\ntext 1" in prompt
    assert "[2] (source: doc.pdf, page 2)\ntext 2" in prompt
    assert prompt.endswith("Question: Q?")


def test_extract_citations_maps_numbers_to_chunks():
    hits = [hit(1), hit(2), hit(3)]
    cited = extract_citations("A [2]. B [1, 2]. C [9].", hits)
    assert [n for n, _ in cited] == [1, 2]
    assert cited[1][1].chunk_id == "doc.pdf#0002"


def test_drop_invalid_citations():
    assert drop_invalid_citations("It is 0x69 [1]. Also listed [15].", 5) == "It is 0x69 [1]. Also listed."
    assert drop_invalid_citations("Both [1, 15] agree.", 5) == "Both [1] agree."
    assert drop_invalid_citations("Fine [2][3].", 5) == "Fine [2][3]."
    assert drop_invalid_citations("No citations.", 5) == "No citations."
