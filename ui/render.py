"""HTML pieces for the Streamlit app. Every piece of document or LLM text is
HTML-escaped before it goes into markup, so nothing in it can inject HTML."""

import html
import re
from pathlib import Path

from src.retrieve import Hit

CSS_PATH = Path(__file__).with_name("style.css")

# Words too common to be worth highlighting in retrieved chunks.
STOPWORDS = set(
    "the a an and or of to in on for is are was were be by with what which how does do did "
    "can when where who why it its this that these those as at from if not default value "
    "there their they you your should".split()
)


def stylesheet() -> str:
    return f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>"


def hero() -> str:
    steps = ["Question", "Dense + BM25 search", "RRF fusion", "Cross-encoder rerank", "Cited answer"]
    chips = '<span class="arrow">→</span>'.join(f'<span class="step">{s}</span>' for s in steps)
    return (
        '<div class="hero"><h1>🤖 Robotics RAG</h1>'
        "<p>Answers about robot hardware and ROS 2 navigation, looked up in the real datasheets "
        "and Nav2 docs, with a citation for every fact. It says so when the documents don't "
        "have the answer.</p>"
        f'<div class="pipeline">{chips}</div></div>'
    )


def answer_markdown(text: str) -> str:
    """The LLM answer as Markdown (for st.markdown), with HTML escaped and
    each [n] citation drawn as a badge."""
    escaped = html.escape(text, quote=False)
    return re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", r'<span class="cite">[\1]</span>', escaped)


def sources(cited: list[tuple[int, Hit]]) -> str:
    chips = "".join(
        f'<span class="source"><b>[{n}]</b> {html.escape(hit.citation)}</span>' for n, hit in cited
    )
    return f'<div class="sources">{chips}</div>'


def query_terms(question: str) -> list[str]:
    """Distinctive words of the question (part codes, parameter names...)."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.\-]*[A-Za-z0-9]|[A-Za-z0-9]", question)
    terms = {w.lower() for w in words if w.lower() not in STOPWORDS and (len(w) >= 3 or any(c.isdigit() for c in w))}
    return sorted(terms, key=len, reverse=True)  # longest first, so "mpu-6050" wins over "6050"


def highlight(text: str, terms: list[str]) -> str:
    """Escape text, then wrap each query term in <mark> (case-insensitive)."""
    escaped = html.escape(text, quote=False)
    if not terms:
        return escaped
    pattern = re.compile(
        r"(?<![A-Za-z0-9])(" + "|".join(re.escape(html.escape(t)) for t in terms) + r")(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    return pattern.sub(r"<mark>\1</mark>", escaped)


def chunk_card(rank: int, hit: Hit, question: str, cited: bool = False, relevant: bool | None = None) -> str:
    """One retrieved chunk. relevant=None means unknown (not a test-set question)."""
    score = f"rerank {hit.rerank_score:.2f}" if hit.rerank_score is not None else f"score {hit.score:.3f}"
    badges = []
    if cited:
        badges.append('<span class="badge cited">cited</span>')
    if relevant:
        badges.append('<span class="badge relevant">✓ contains the answer</span>')
    classes = "chunk" + (" relevant" if relevant else " cited" if cited else "")
    # Whitespace collapsed: PDF text is full of hard line breaks, and a blank
    # line inside the HTML would end it early when Markdown renders it.
    body = highlight(" ".join(hit.text.split()), query_terms(question))
    return (
        f'<div class="{classes}"><div class="head"><span class="rank">#{rank}</span>'
        f'<span class="where">{html.escape(hit.citation)}</span>{"".join(badges)}'
        f'<span class="score">{score}</span></div><div class="text">{body}</div></div>'
    )


def mode_head(title: str, subtitle: str) -> str:
    return f'<div class="mode-head"><h4>{html.escape(title)}</h4><p>{html.escape(subtitle)}</p></div>'


def tiles(items: list[dict]) -> str:
    """KPI row. Each item: label, value, optional from (old value), note, delta."""
    cells = []
    for item in items:
        frm = f'<span class="from">{item["from"]} → </span>' if item.get("from") else ""
        delta = f' <span class="delta">{item["delta"]}</span>' if item.get("delta") else ""
        note = f'<div class="note">{item["note"]}{delta}</div>' if item.get("note") or delta else ""
        cells.append(
            f'<div class="tile"><div class="label">{item["label"]}</div>'
            f'<div class="value">{frm}{item["value"]}</div>{note}</div>'
        )
    return f'<div class="tiles">{"".join(cells)}</div>'
