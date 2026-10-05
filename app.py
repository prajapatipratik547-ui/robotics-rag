"""Streamlit UI: ask questions in a chat, compare retrieval modes side by side,
and see the evaluation results.

Run with:  streamlit run app.py
"""

import os
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from datetime import date

import streamlit as st

# On Streamlit Community Cloud, keys like GROQ_API_KEY live in st.secrets.
# Copy them into the environment before config.py reads it. Locally there is
# no secrets file and .env is used instead.
try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, (str, int, float)):
            os.environ.setdefault(_key, str(_value))
except Exception:  # no secrets file: running locally
    pass

import config
from src.generate import NOT_FOUND, Answer, QuotaExceeded, extract_citations, generate
from src.retrieve import Hit, get_client, get_dense_model, get_reranker, get_sparse_model, retrieve
from src.testset import CHUNKS_PATH, TESTSET_PATH, load_jsonl, relevant_chunk_ids
from ui import evaluation, render

EXAMPLES = [
    "What is the distance range of the RPLiDAR A1M8?",
    "What is the I2C address of the MPU-6050 when AD0 is high?",
    "What is the default inflation_radius in Nav2?",
    "What motion models does the MPPI controller support?",
]
MODE_LABELS = {
    "hybrid": "Hybrid (dense + BM25, RRF)",
    "dense": "Dense only (semantic)",
    "sparse": "Sparse only (BM25 keywords)",
}
# The three configurations the evaluation compares.
COMPARE_CONFIGS = [
    ("Dense only", "Semantic search: the Stage 2 baseline", "dense", False),
    ("Hybrid", "Dense + BM25 keywords, fused with RRF", "hybrid", False),
    ("Hybrid + rerank", "Then a cross-encoder reorders the top 20", "hybrid", True),
]
COMPARE_DEFAULT = "What is the default motion model of the Smac Hybrid-A* planner?"
# The index exists once ingestion has written the collection and the chunk cache.
INDEX_PATHS = [config.QDRANT_PATH / "collection" / config.COLLECTION_NAME, CHUNKS_PATH]
ANSWER_STORE_SIZE = 1000

st.set_page_config(page_title="Robotics RAG", page_icon="🤖", layout="wide")


# --- one-time setup ----------------------------------------------------------


@st.cache_resource
def build_lock() -> threading.Lock:
    return threading.Lock()


def ensure_index() -> None:
    """On a fresh host (e.g. the deployed app), download the documents and build
    the index on first start. Runs the two scripts as subprocesses, with their
    output shown live. Checked on disk, before the database is opened."""
    if all(p.exists() for p in INDEX_PATHS):
        return
    with build_lock():  # several visitors arriving at once build it only once
        if all(p.exists() for p in INDEX_PATHS):
            return
        with st.status("First start: building the search index (about 3-5 minutes)...", expanded=True) as status:
            log = st.empty()
            lines: list[str] = []
            for label, script, must_succeed in [
                ("Downloading the 32 source documents", "data/download_corpus.py", False),
                ("Chunking and embedding them", "src/ingest.py", True),
            ]:
                status.update(label=f"{label}...")
                process = subprocess.Popen(
                    [sys.executable, "-u", str(config.ROOT_DIR / script)],
                    cwd=config.ROOT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                )
                for line in process.stdout:
                    lines.append(line.rstrip())
                    log.code("\n".join(lines[-12:]), language=None)
                # A few failed downloads are fine; the index just has fewer documents.
                if process.wait() != 0 and must_succeed:
                    status.update(label="Building the index failed", state="error")
                    st.stop()
            status.update(label="Search index ready", state="complete", expanded=False)


@st.cache_resource(show_spinner="Loading the models (first run only)...")
def load_pipeline() -> None:
    """Load everything once per server process, so the first question isn't slow."""
    if not get_client().collection_exists(config.COLLECTION_NAME):
        raise RuntimeError(f"Collection {config.COLLECTION_NAME!r} not found")
    get_dense_model()
    get_sparse_model()
    get_reranker()


@st.cache_data(show_spinner=False)
def load_testset() -> dict[str, dict]:
    """Test-set questions -> their item, with the IDs of the answer-bearing chunks."""
    chunks = load_jsonl(CHUNKS_PATH)
    items = {}
    for item in load_jsonl(TESTSET_PATH):
        try:
            items[item["question"]] = {**item, "relevant": relevant_chunk_ids(item, chunks)}
        except ValueError:  # e.g. a document failed to download on this host
            pass
    return items


# --- answering, shared across visitors ---------------------------------------------


@st.cache_data(show_spinner=False, max_entries=500)
def cached_retrieve(question: str, mode: str, use_rerank: bool) -> tuple[list[Hit], float]:
    start = time.perf_counter()
    hits = retrieve(question, config.TOP_K, mode, use_rerank)
    return hits, time.perf_counter() - start


@st.cache_resource
def answer_store() -> dict:
    """Answers already given, keyed by question + retrieved chunks, shared by all
    visitors; a repeat question costs no LLM call. Plus today's LLM call count."""
    return {"answers": OrderedDict(), "day": None, "llm_calls": 0, "lock": threading.Lock()}


def get_answer(question: str, hits: list[Hit]) -> tuple[Answer | None, float]:
    """The cited answer, or None when today's public-demo LLM limit is used up."""
    store = answer_store()
    key = (question.strip().lower(), config.LLM_PROVIDER, tuple(h.chunk_id for h in hits))
    with store["lock"]:
        if key in store["answers"]:
            text = store["answers"][key]
            return Answer(question, text, hits, extract_citations(text, hits)), 0.0
        if store["day"] != date.today():
            store["day"], store["llm_calls"] = date.today(), 0
        if config.DAILY_LLM_LIMIT and store["llm_calls"] >= config.DAILY_LLM_LIMIT:
            return None, 0.0
        store["llm_calls"] += 1
    start = time.perf_counter()
    answer = generate(question, hits)
    with store["lock"]:
        store["answers"][key] = answer.text
        if len(store["answers"]) > ANSWER_STORE_SIZE:
            store["answers"].popitem(last=False)
    return answer, time.perf_counter() - start


# --- rendering ---------------------------------------------------------------------


def show_chunks(question: str, hits: list[Hit], cited: set[int], relevant: set[str] | None) -> None:
    for i, hit in enumerate(hits, start=1):
        is_relevant = None if relevant is None else hit.chunk_id in relevant
        st.html(render.chunk_card(i, hit, question, cited=i in cited, relevant=is_relevant))


def show_answer(turn: dict, key: str, passages: bool = True) -> None:
    """One assistant turn: answer card, source chips, and (optionally) the
    retrieved chunks in an expander."""
    answer: Answer | None = turn["answer"]
    with st.container(key=f"answer-{key}"):
        if answer is None:
            st.warning("Today's demo limit for LLM answers is used up. Here is what retrieval found.")
        elif answer.text == NOT_FOUND:
            st.markdown(f"🤷 {answer.text}")
        else:
            st.markdown(render.answer_markdown(answer.text), unsafe_allow_html=True)
        if answer and answer.cited:
            st.html(render.sources(answer.cited))
        llm = f" · LLM {turn['generation_s']:.2f} s" if turn["generation_s"] else ""
        st.html(f'<div class="meta">{MODE_LABELS[turn["mode"]]}{" + rerank" if turn["rerank"] else ""}'
                f" · retrieval {turn['retrieval_s']:.2f} s{llm}</div>")
    if passages:
        cited = {n for n, _ in answer.cited} if answer else set()
        with st.expander(f"Retrieved passages (top {len(turn['hits'])}), as given to the LLM"):
            show_chunks(turn["question"], turn["hits"], cited, None)


def ask(question: str, mode: str, use_rerank: bool) -> dict:
    hits, retrieval_s = cached_retrieve(question, mode, use_rerank)
    answer, generation_s = get_answer(question, hits)
    return {"question": question, "mode": mode, "rerank": use_rerank, "hits": hits,
            "answer": answer, "retrieval_s": retrieval_s, "generation_s": generation_s}


def show_error(error: Exception) -> None:
    """A plain message for visitors; the provider's full error goes to the server log."""
    if isinstance(error, QuotaExceeded):
        print(error, file=sys.stderr)
        st.warning("The free LLM tier is busy right now (rate limit). Wait a minute and ask again.")
    else:
        st.error(str(error))


# --- tabs ----------------------------------------------------------------------------


def chat_tab(mode: str, use_rerank: bool) -> None:
    history = st.session_state.setdefault("history", [])
    if not history:
        st.caption("Try an example, or type your own question below.")
        for column, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
            if column.button(example, width="stretch"):
                st.session_state.pending = example

    conversation = st.container()  # filled before the input box, so new turns appear above it
    prompt = st.chat_input("Ask about a sensor, board, motor driver, or Nav2...")
    question = (prompt or st.session_state.pop("pending", "") or "").strip()

    with conversation:
        for i, turn in enumerate(history):
            with st.chat_message("user"):
                st.markdown(turn["question"])
            with st.chat_message("assistant", avatar="🤖"):
                show_answer(turn, str(i))
        if question:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Searching the documents and writing an answer..."):
                    try:
                        turn = ask(question, mode, use_rerank)
                    except (QuotaExceeded, RuntimeError) as e:
                        show_error(e)
                        return
                show_answer(turn, str(len(history)))
            history.append(turn)

    if history and st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()


def compare_tab() -> None:
    st.markdown(
        "Run one question through all three retrieval setups from the evaluation. For questions from "
        "the test set, passages that contain the answer are marked **✓**."
    )
    testset = load_testset()
    options = ["(type your own below)"] + list(testset)
    # Default to a question where the modes visibly differ (dense misses it).
    default = options.index(COMPARE_DEFAULT) if COMPARE_DEFAULT in options else 0
    choice = st.selectbox("Pick a test-set question", options, index=default)
    own = st.text_input("Or your own question", placeholder="e.g. What is the stall torque of the MG996R?")
    question = (own or (choice if choice in testset else "")).strip()
    if not question or not st.button("Compare the three modes", type="primary"):
        return

    relevant = set(testset[question]["relevant"]) if question in testset else None
    for column, (title, subtitle, mode, use_rerank) in zip(st.columns(3), COMPARE_CONFIGS):
        with column:
            st.html(render.mode_head(title, subtitle))
            with st.spinner("Working..."):
                try:
                    turn = ask(question, mode, use_rerank)
                except (QuotaExceeded, RuntimeError) as e:
                    show_error(e)
                    continue
            if relevant is not None:
                ranks = [i for i, h in enumerate(turn["hits"], start=1) if h.chunk_id in relevant]
                st.html(f'<span class="badge relevant">answer found at #{ranks[0]}</span>' if ranks
                        else '<span class="badge miss">answer not retrieved</span>')
            show_answer(turn, f"cmp-{mode}-{use_rerank}", passages=False)
            # The ranked passages show directly here: the ranking is the point of this tab.
            cited = {n for n, _ in turn["answer"].cited} if turn["answer"] else set()
            show_chunks(question, turn["hits"], cited, relevant)


def sidebar() -> tuple[str, bool]:
    with st.sidebar:
        st.header("Retrieval settings")
        mode = st.radio("Mode (Ask tab)", list(MODE_LABELS), format_func=MODE_LABELS.get)
        use_rerank = st.checkbox(
            "Rerank with cross-encoder",
            value=config.DEFAULT_RERANK,
            help=f"Fetch {config.RERANK_CANDIDATES} candidates, let `{config.RERANK_MODEL}` reorder them, "
            f"keep the best {config.TOP_K}.",
        )
        st.divider()
        st.markdown("**How it works**")
        st.caption(
            "Each question is searched two ways: by meaning (dense embeddings) and by exact words "
            "(BM25), which catches part codes like `A1M8`. The two result lists are merged with "
            "Reciprocal Rank Fusion, a cross-encoder reranks the top 20, and the LLM answers from "
            "the best 5 only, citing each one."
        )
        model = config.GROQ_MODEL if config.LLM_PROVIDER == "groq" else config.GEMINI_MODEL
        st.caption(f"Answers: `{model}` · Reranker: `{config.RERANK_MODEL}`")
        st.markdown("[Source code on GitHub](https://github.com/prajapatipratik547-ui/robotics-rag)")
    return mode, use_rerank


def main() -> None:
    st.html(render.stylesheet())
    st.html(render.hero())
    mode, use_rerank = sidebar()

    ensure_index()
    try:
        load_pipeline()
    except Exception as e:  # the most common causes get a specific hint
        message = str(e)
        if "already accessed by another instance" in message:
            st.error(
                "The vector store is in use by another process (the CLI or `eval/run_eval.py`). "
                "Qdrant's local mode allows one process at a time; stop the other one and reload."
            )
        elif "not found" in message.lower():
            st.error("No documents are indexed yet. Run `python src/ingest.py` first, then reload.")
        else:
            st.error(f"Could not load the pipeline: {message}")
        st.stop()

    ask_tab, compare, results = st.tabs(["💬 Ask", "⚖️ Compare modes", "📊 Evaluation"])
    with ask_tab:
        chat_tab(mode, use_rerank)
    with compare:
        compare_tab()
    with results:
        evaluation.show()


main()
