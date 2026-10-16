"""Loading and chunking. Downloads the bge-small tokenizer (~1 MB) on first run."""

import pytest
from tokenizers import Tokenizer

import config
from src.ingest import chunk_text, split_markdown_sections, split_units


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained(config.DENSE_MODEL)


def n_tokens(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def test_markdown_sections_ignore_headings_in_code_fences():
    text = "# Setup\nInstall it.\n```bash\n# not a heading\nsudo apt install x\n```\n## Params\nDefault: 1"
    sections = split_markdown_sections(text, "guide.md")
    assert [s.section for s in sections] == ["Setup", "Params"]
    assert "# not a heading" in sections[0].text


def test_split_units_keeps_paragraph_and_sentence_separators():
    units = split_units("First one. Second one.\n\nNew paragraph.")
    assert units == [("", "First one."), (" ", "Second one."), ("\n\n", "New paragraph.")]


def test_chunks_respect_the_token_limit_and_overlap(tokenizer):
    sentences = [f"Sentence number {i} says the robot drives at speed {i} metres per second." for i in range(120)]
    chunks = chunk_text(" ".join(sentences), tokenizer, max_tokens=100, overlap=25)
    assert len(chunks) > 5
    assert all(n_tokens(tokenizer, c) <= 100 for c in chunks)
    # Consecutive chunks share their boundary sentences, so no fact is cut in two.
    for previous, current in zip(chunks, chunks[1:]):
        assert current.split(". ")[0] in previous


def test_an_overlong_unit_is_split_on_word_boundaries(tokenizer):
    table_row = " ".join(f"value{i}" for i in range(400))  # one "sentence", far over the limit
    chunks = chunk_text(table_row, tokenizer, max_tokens=60, overlap=10)
    assert len(chunks) > 1
    assert all(n_tokens(tokenizer, c) <= 60 for c in chunks)
    assert all(word.startswith("value") for c in chunks for word in c.split())
