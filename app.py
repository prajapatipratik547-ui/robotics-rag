"""Streamlit UI: ask a question, get a cited answer and see what was retrieved.

Run with:  streamlit run app.py
"""

import time

import streamlit as st

import config
from src.generate import NOT_FOUND, Answer, QuotaExceeded, generate
from src.retrieve import get_client, get_dense_model, get_reranker, get_sparse_model, retrieve

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

st.set_page_config(page_title="Robotics RAG", page_icon="🤖", layout="wide")


@st.cache_resource(show_spinner="Loading the models and the vector store (first run only)...")
def load_pipeline() -> None:
    """Load everything once per server process, so the first question isn't slow."""
    if not get_client().collection_exists(config.COLLECTION_NAME):
        raise RuntimeError(f"Collection {config.COLLECTION_NAME!r} not found")
    get_dense_model()
    get_sparse_model()
    get_reranker()


@st.cache_data(show_spinner=False, max_entries=200)
def answer_question(question: str, mode: str, use_rerank: bool) -> tuple[Answer, float, float]:
    """Retrieve + generate, cached so re-asking (or any page rerun) doesn't call
    the LLM again. Errors aren't cached, so a failed question can be retried."""
    start = time.perf_counter()
    hits = retrieve(question, config.TOP_K, mode, use_rerank)
    retrieved_at = time.perf_counter()
    answer = generate(question, hits)
    return answer, retrieved_at - start, time.perf_counter() - retrieved_at


def show_answer(answer: Answer, retrieval_s: float, generation_s: float) -> None:
    st.subheader("Answer")
    if answer.text == NOT_FOUND:
        st.warning(answer.text)
    else:
        st.markdown(answer.text)
    st.caption(f"Retrieval {retrieval_s:.2f} s · LLM {generation_s:.2f} s")

    if answer.cited:
        st.markdown("**Sources**")
        for n, hit in answer.cited:
            st.markdown(f"[{n}] `{hit.citation}`")

    cited_numbers = {n for n, _ in answer.cited}
    with st.expander(f"Retrieved chunks (top {len(answer.hits)}), as given to the LLM"):
        for i, hit in enumerate(answer.hits, start=1):
            score = f"rerank score {hit.rerank_score:.2f}" if hit.rerank_score is not None else f"score {hit.score:.3f}"
            mark = " · **cited**" if i in cited_numbers else ""
            # Code formatting: section names like "<inflation layer>.x" would otherwise read as HTML.
            st.markdown(f"**[{i}]** `{hit.citation}` · {score}{mark}")
            st.code(hit.text, language=None, wrap_lines=True)


def main() -> None:
    with st.sidebar:
        st.header("Retrieval settings")
        mode = st.radio("Mode", list(MODE_LABELS), format_func=MODE_LABELS.get)
        use_rerank = st.checkbox(
            "Rerank with cross-encoder",
            value=config.DEFAULT_RERANK,
            help=f"Fetch {config.RERANK_CANDIDATES} candidates, let bge-reranker-base reorder them, "
            f"keep the best {config.TOP_K}. More precise, but adds a few seconds on CPU.",
        )
        st.divider()
        st.caption(
            "Measured on a 26-question test set (eval/results.md): hybrid retrieval found the "
            "answer-bearing chunk in the top 5 for 96% of questions, vs 88% for dense only."
        )
        st.caption(f"Answers by `{config.GROQ_MODEL if config.LLM_PROVIDER == 'groq' else config.GEMINI_MODEL}`.")

    st.title("Robotics RAG")
    st.write(
        "Ask about the RPLiDAR A1M8, Raspberry Pi 4, Arduino Mega 2560, MPU-6050, NEO-6M, "
        "L298N, HC-SR04, MG996R, or ROS 2 Nav2. Answers come only from those documents, with citations."
    )

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

    st.caption("Try an example:")
    for column, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
        if column.button(example, width="stretch"):
            st.session_state.question = example
    question = st.text_input(
        "Your question", key="question", placeholder="e.g. What is the stall torque of the MG996R?"
    ).strip()
    if not question:
        return

    with st.spinner("Retrieving and answering..."):
        try:
            answer, retrieval_s, generation_s = answer_question(question, mode, use_rerank)
        except (QuotaExceeded, RuntimeError) as e:
            st.error(str(e))
            return
    show_answer(answer, retrieval_s, generation_s)


main()
