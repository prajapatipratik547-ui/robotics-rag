"""Stage 1 ingestion: load -> chunk -> embed -> store.

Reads every PDF / Markdown / HTML file in data/raw/, splits it into
overlapping chunks, embeds each chunk twice (dense + BM25 sparse), and
stores both vectors on the same Qdrant point.

Run:  python src/ingest.py
"""

import json
import re
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf
from bs4 import BeautifulSoup, Comment
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
from tokenizers import Tokenizer

import config

UPSERT_BATCH_SIZE = 64


@dataclass
class Section:
    """A contiguous piece of a document under one heading (or one PDF page)."""

    source_file: str
    section: str
    text: str
    page: int | None = None


# ---------------------------------------------------------------------------
# 1. Load: turn each file type into a list of Sections
# ---------------------------------------------------------------------------

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def split_markdown_sections(text: str, source_file: str) -> list[Section]:
    """Split Markdown-style text on '#' headings. Headings inside ``` code
    fences are ignored (ROS docs have bash comments that start with '#')."""
    sections: list[Section] = []
    heading = "(intro)"
    lines: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            sections.append(Section(source_file, heading, body))

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        match = None if in_fence else HEADING_RE.match(line)
        if match:
            flush()
            heading, lines = match.group(2).strip(), []
        else:
            lines.append(line)
    flush()
    return sections


def load_markdown(path: Path) -> list[Section]:
    return split_markdown_sections(path.read_text(encoding="utf-8"), path.name)


def load_html(path: Path) -> list[Section]:
    """Strip page chrome, turn <h1>-<h6> into '#' lines, then reuse the
    Markdown splitter so both formats get identical section handling."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    # Comments must go first: the whitespace pass below would otherwise turn
    # them into ordinary visible text (e.g. a theme's license header).
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    # Documentation sites mark the article itself; reading only that drops
    # sidebars, banners and "skip to content" links. Plain pages fall back
    # to the whole <body>.
    root = soup.find("article") or soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup
    for tag in root(["script", "style", "nav", "footer", "header", "noscript", "svg", "button", "form", "aside"]):
        tag.decompose()
    for permalink in root.select("a.headerlink"):  # the '¶' anchor next to headings
        permalink.decompose()

    # Collapse source-code line wrapping inside text, except in <pre> blocks.
    for string in root.find_all(string=True):
        if not string.find_parent("pre"):
            string.replace_with(re.sub(r"\s+", " ", string))

    for h in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        level = int(h.name[1])
        h.replace_with(f"\n\n{'#' * level} {h.get_text(' ', strip=True)}\n\n")
    for block in root.find_all(["p", "pre", "table", "ul", "ol"]):
        block.append("\n\n")
    for line_break in root.find_all(["li", "tr", "br"]):
        line_break.append("\n")

    text = root.get_text()
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return split_markdown_sections(text, path.name)


def load_pdf(path: Path) -> list[Section]:
    """One Section per page. Datasheet PDFs rarely have reliable headings,
    so the page number is the section label (and makes citations precise)."""
    sections: list[Section] = []
    with pymupdf.open(path) as doc:
        for page_number, page in enumerate(doc, start=1):
            # get_text("blocks") keeps PyMuPDF's paragraph grouping; block[6]
            # is 0 for text blocks (1 would be an image).
            paragraphs = [
                " ".join(block[4].split())
                for block in page.get_text("blocks")
                if block[6] == 0 and block[4].strip()
            ]
            body = "\n\n".join(paragraphs)
            if body:
                sections.append(Section(path.name, f"page {page_number}", body, page_number))
    return sections


LOADERS = {".pdf": load_pdf, ".md": load_markdown, ".html": load_html, ".htm": load_html}


def load_documents(raw_dir: Path) -> list[Section]:
    sections: list[Section] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        loader = LOADERS.get(path.suffix.lower())
        if loader is None:
            print(f"  skip (unsupported type): {path.name}")
            continue
        file_sections = loader(path)
        print(f"  loaded {path.name}: {len(file_sections)} sections")
        sections.extend(file_sections)
    return sections


# ---------------------------------------------------------------------------
# 2. Chunk: pack sentences into ~400-token chunks with ~100 tokens of overlap
# ---------------------------------------------------------------------------

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


def split_units(text: str) -> list[tuple[str, str]]:
    """Break text into sentence-sized units, remembering the separator that
    came before each one, so chunks can be re-joined with their original
    paragraph and line structure intact."""
    units: list[tuple[str, str]] = []
    for p_index, paragraph in enumerate(text.split("\n\n")):
        for l_index, line in enumerate(paragraph.splitlines()):
            for s_index, sentence in enumerate(SENTENCE_RE.split(line.strip())):
                if not sentence:
                    continue
                if s_index > 0:
                    sep = " "
                elif l_index > 0:
                    sep = "\n"
                else:
                    sep = "\n\n" if p_index > 0 else ""
                units.append((sep, sentence))
    return units


def hard_split(text: str, tokenizer: Tokenizer, max_tokens: int, overlap: int) -> list[str]:
    """Fallback for a single unit longer than max_tokens (e.g. a huge table
    row): slice it on token boundaries, using character offsets so the
    original text (with its casing) is preserved."""
    offsets = tokenizer.encode(text, add_special_tokens=False).offsets
    pieces = []
    for start in range(0, len(offsets), max_tokens - overlap):
        window = offsets[start : start + max_tokens]
        begin, end = window[0][0], window[-1][1]
        # Snap both ends to whitespace so a word is never cut in half.
        snapped_begin, snapped_end = begin, end
        while 0 < snapped_begin < end and not text[snapped_begin - 1].isspace():
            snapped_begin += 1
        while snapped_end < len(text) and not text[snapped_end].isspace():
            snapped_end += 1
        if snapped_begin < end:
            begin = snapped_begin
        pieces.append(text[begin:snapped_end].strip())
        if start + max_tokens >= len(offsets):
            break
    return pieces


def chunk_text(text: str, tokenizer: Tokenizer, max_tokens: int, overlap: int) -> list[str]:
    """Greedily add sentences to the current chunk until the next one would
    overflow max_tokens. Then start a new chunk that begins with the last
    ~overlap tokens' worth of sentences from the previous one, so a fact
    that straddles a boundary is fully present in at least one chunk."""
    units: list[tuple[str, str, int]] = []
    for sep, unit in split_units(text):
        n_tokens = len(tokenizer.encode(unit, add_special_tokens=False).ids)
        if n_tokens <= max_tokens:
            units.append((sep, unit, n_tokens))
        else:
            for i, piece in enumerate(hard_split(unit, tokenizer, max_tokens, overlap)):
                units.append((sep if i == 0 else " ", piece, max_tokens))

    def join(parts: list[tuple[str, str, int]]) -> str:
        return "".join(sep + unit for sep, unit, _ in parts).strip()

    chunks: list[str] = []
    current: list[tuple[str, str, int]] = []
    current_len = 0
    for sep, unit, n_tokens in units:
        if current and current_len + n_tokens > max_tokens:
            chunks.append(join(current))
            kept: list[tuple[str, str, int]] = []
            kept_len = 0
            for part in reversed(current):
                if kept_len + part[2] > overlap:
                    break
                kept.insert(0, part)
                kept_len += part[2]
            if kept_len + n_tokens > max_tokens:
                kept, kept_len = [], 0
            current, current_len = kept, kept_len
        current.append((sep, unit, n_tokens))
        current_len += n_tokens
    if current:
        chunks.append(join(current))
    return chunks


def doc_title(source_file: str) -> str:
    """'rplidar_a1m8_datasheet.pdf' -> 'rplidar a1m8 datasheet'."""
    return re.sub(r"[_\-]+", " ", Path(source_file).stem)


def embedding_input(chunk: dict) -> str:
    """What actually gets embedded: the chunk prefixed with its document title
    and section. A datasheet page saying 'Scan frequency: 5.5 Hz' often never
    names the part; the prefix puts 'rplidar a1m8' into both vectors."""
    return f"{doc_title(chunk['source_file'])} | {chunk['section']}\n\n{chunk['text']}"


def build_chunks(sections: list[Section], tokenizer: Tokenizer) -> list[dict]:
    chunks: list[dict] = []
    counters: dict[str, int] = {}
    for sec in sections:
        for text in chunk_text(sec.text, tokenizer, config.CHUNK_MAX_TOKENS, config.CHUNK_OVERLAP_TOKENS):
            index = counters.get(sec.source_file, 0)
            counters[sec.source_file] = index + 1
            chunks.append(
                {
                    "chunk_id": f"{sec.source_file}#{index:04d}",
                    "source_file": sec.source_file,
                    "section": sec.section,
                    "page": sec.page,
                    "text": text,
                }
            )
    return chunks


# ---------------------------------------------------------------------------
# 3 + 4. Embed and store: one point per chunk, with BOTH named vectors
# ---------------------------------------------------------------------------


def reset_store() -> None:
    """Wipe the local Qdrant folder so every run rebuilds from scratch and
    never keeps stale chunks from deleted or edited files.

    This must happen before a client is opened. client.delete_collection()
    is not enough on Windows: local mode keeps the collection's SQLite file
    open, the folder delete silently fails, and recreating the collection
    brings the old points back."""
    if not config.QDRANT_PATH.exists():
        return
    try:
        shutil.rmtree(config.QDRANT_PATH)
    except PermissionError:
        print(f"Cannot reset {config.QDRANT_PATH}: it is open in another process "
              "(e.g. the Streamlit app or another Python session). Close it and re-run.")
        sys.exit(1)


def create_collection(client: QdrantClient, dense_dim: int) -> None:
    client.create_collection(
        config.COLLECTION_NAME,
        vectors_config={
            config.DENSE_VECTOR_NAME: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            # FastEmbed's BM25 model only emits term-frequency weights. The
            # IDF half of BM25 depends on the whole corpus, so Qdrant
            # computes it at query time when this modifier is set.
            config.SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF),
        },
    )


def embed_and_store(client: QdrantClient, chunks: list[dict]) -> None:
    dense_model = TextEmbedding(config.DENSE_MODEL, cache_dir=str(config.MODEL_CACHE_DIR))
    sparse_model = SparseTextEmbedding(config.SPARSE_MODEL, cache_dir=str(config.MODEL_CACHE_DIR))

    inputs = [embedding_input(c) for c in chunks]
    dense_vectors = list(dense_model.embed(inputs, batch_size=config.EMBED_BATCH_SIZE))
    sparse_vectors = list(sparse_model.embed(inputs, batch_size=config.EMBED_BATCH_SIZE))

    create_collection(client, dense_dim=len(dense_vectors[0]))

    points = [
        models.PointStruct(
            # Deterministic id: the same chunk always maps to the same point.
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"])),
            vector={
                config.DENSE_VECTOR_NAME: dense.tolist(),
                config.SPARSE_VECTOR_NAME: models.SparseVector(
                    indices=sparse.indices.tolist(), values=sparse.values.tolist()
                ),
            },
            payload=chunk,
        )
        for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors)
    ]
    for start in range(0, len(points), UPSERT_BATCH_SIZE):
        client.upsert(config.COLLECTION_NAME, points=points[start : start + UPSERT_BATCH_SIZE])


def verify(client: QdrantClient, expected: int) -> None:
    """DoD check: the collection holds exactly the chunks just stored (no
    stale leftovers) and every point carries a dense AND a sparse vector."""
    count = client.count(config.COLLECTION_NAME, exact=True).count
    both, offset = 0, None
    sample = None
    while True:
        points, offset = client.scroll(
            config.COLLECTION_NAME, limit=256, offset=offset, with_vectors=True, with_payload=True
        )
        for point in points:
            dense = point.vector.get(config.DENSE_VECTOR_NAME)
            sparse = point.vector.get(config.SPARSE_VECTOR_NAME)
            if dense and sparse and sparse.indices:
                both += 1
                sample = sample or (point, dense, sparse)
        if offset is None:
            break

    print(f"\nVerify: collection '{config.COLLECTION_NAME}' has {count} points; "
          f"{both}/{count} carry both dense and sparse vectors.")
    if sample:
        point, dense, sparse = sample
        print(f"  sample {point.payload['chunk_id']}: dense dim={len(dense)}, "
              f"sparse non-zero terms={len(sparse.indices)}")
    if count != expected:
        print(f"FAIL: expected {expected} points (one per chunk), found {count}; stale points survived the reset.")
        sys.exit(1)
    if count == 0 or both != count:
        print("FAIL: collection is empty or some points are missing a vector.")
        sys.exit(1)
    print("PASS")


def main() -> None:
    if not config.RAW_DATA_DIR.exists() or not any(
        p.suffix.lower() in config.SUPPORTED_EXTENSIONS for p in config.RAW_DATA_DIR.rglob("*")
    ):
        print(f"No PDF/MD/HTML files found in {config.RAW_DATA_DIR}. Add documents and re-run.")
        sys.exit(1)

    print(f"Loading documents from {config.RAW_DATA_DIR}")
    sections = load_documents(config.RAW_DATA_DIR)

    tokenizer = Tokenizer.from_pretrained(config.DENSE_MODEL)
    tokenizer.no_truncation()
    chunks = build_chunks(sections, tokenizer)
    sizes = [len(tokenizer.encode(c["text"], add_special_tokens=False).ids) for c in chunks]
    print(f"\nChunked {len(sections)} sections into {len(chunks)} chunks "
          f"(tokens per chunk: min {min(sizes)}, avg {sum(sizes) // len(sizes)}, max {max(sizes)})")

    # Cache the chunks so they can be inspected by eye.
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = config.PROCESSED_DATA_DIR / "chunks.jsonl"
    with cache.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Wrote {cache.relative_to(config.ROOT_DIR)}")

    print("\nEmbedding (dense + BM25) and storing in Qdrant...")
    reset_store()
    client = QdrantClient(path=str(config.QDRANT_PATH))
    try:
        embed_and_store(client, chunks)
        print(f"Stored {len(chunks)} chunks.")
        verify(client, expected=len(chunks))
    finally:
        client.close()


if __name__ == "__main__":
    main()
