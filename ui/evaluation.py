"""The Evaluation tab: the Stage 5 numbers from eval/results.json."""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from ui import render

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
RESULTS_JSON = EVAL_DIR / "results.json"
RERANKERS_DIR = EVAL_DIR / "rerankers"

MODE_NAMES = {"dense": "Dense only", "hybrid": "Hybrid", "hybrid+rerank": "Hybrid + rerank"}
# Emphasis form: the baseline is gray, the two hybrid modes share one blue hue
# (dark-theme ordinal steps, validated against the #151b2b chart surface).
MODE_COLORS = {"Dense only": "#898781", "Hybrid": "#86b6ef", "Hybrid + rerank": "#3987e5"}
METRIC_NAMES = {
    "hit": "Hit@5",
    "mrr": "MRR",
    "recall": "Context recall",
    "precision": "Context precision",
    "faithfulness": "Faithfulness",
    "accuracy": "Answer accuracy",
}


@st.cache_data(show_spinner=False)
def _load(path: str, mtime: float) -> dict:  # mtime in the key: reload when the file changes
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_json(path: Path) -> dict | None:
    return _load(str(path), path.stat().st_mtime) if path.exists() else None


def bar_chart(summary: dict, metric: str) -> alt.Chart:
    data = pd.DataFrame(
        {"Mode": MODE_NAMES[m], "Score": summary[m][metric]} for m in MODE_NAMES if summary[m].get(metric) is not None
    )
    is_percent = metric == "hit"
    fmt = ".0%" if is_percent else ".3f"  # value labels and tooltips
    tick_fmt = ".0%" if is_percent else ".1f"
    base = alt.Chart(data, title=METRIC_NAMES[metric]).encode(
        y=alt.Y("Mode:N", sort=list(MODE_NAMES.values()), title=None, axis=alt.Axis(labelLimit=200)),
        x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1.12]), title=None,
                axis=alt.Axis(format=tick_fmt, values=[0, 0.25, 0.5, 0.75, 1])),
    )
    bars = base.mark_bar(cornerRadiusEnd=4, size=22).encode(
        color=alt.Color("Mode:N", scale=alt.Scale(domain=list(MODE_COLORS), range=list(MODE_COLORS.values())), legend=None),
        tooltip=[alt.Tooltip("Mode:N"), alt.Tooltip("Score:Q", format=fmt, title=METRIC_NAMES[metric])],
    )
    labels = base.mark_text(align="left", dx=6, color="#e6e9f0", fontSize=13).encode(text=alt.Text("Score:Q", format=fmt))
    return (
        (bars + labels)
        .properties(height=alt.Step(40))  # one 40px band per bar: 22px bar + visible gap
        .configure(background="#151b2b")
        .configure_view(stroke=None)
        .configure_axis(labelColor="#b4bccb", gridColor="rgba(255,255,255,0.07)", domainColor="#383835", tickColor="#383835")
        .configure_title(color="#e6e9f0", anchor="start", fontSize=14)
    )


def show() -> None:
    results = load_json(RESULTS_JSON)
    if results is None:
        st.info("No evaluation results yet. Run `python eval/run_eval.py --retrieval-only` to create them.")
        return
    summary = results["summary"]
    dense, hybrid, rerank = summary["dense"], summary["hybrid"], summary["hybrid+rerank"]
    n = len(results["per_question"])

    st.markdown(
        f"Measured on **{n} hand-built questions**, each with a confirmed answer and the exact passages in "
        "the documents that contain it. Everything below comes from `eval/results.json`, written by "
        "`python eval/run_eval.py`."
    )
    st.html(
        render.tiles(
            [
                {
                    "label": "Retrieval accuracy (Hit@5)",
                    "from": f"{dense['hit']:.0%}",
                    "value": f"{rerank['hit']:.0%}",
                    "note": "answer-bearing chunk in the top 5",
                    "delta": f"+{(rerank['hit'] - dense['hit']) * 100:.0f} pts",
                },
                {
                    "label": "Ranking quality (MRR)",
                    "from": f"{dense['mrr']:.3f}",
                    "value": f"{rerank['mrr']:.3f}",
                    "note": "1.0 = always ranked first",
                    "delta": f"+{rerank['mrr'] - dense['mrr']:.3f}",
                },
                {
                    "label": "Context recall",
                    "from": f"{dense['recall']:.2f}",
                    "value": f"{max(hybrid['recall'], rerank['recall']):.2f}",
                    "note": "share of answer chunks retrieved",
                },
                {
                    "label": "Latency per question",
                    "value": f"{hybrid['latency_s']:.2f} s",
                    "note": f"hybrid; reranking adds ~{rerank['latency_s'] - hybrid['latency_s']:.1f} s on CPU",
                },
            ]
        )
    )

    left, right = st.columns(2)
    left.altair_chart(bar_chart(summary, "hit"), width="stretch")
    right.altair_chart(bar_chart(summary, "mrr"), width="stretch")
    st.caption(
        "Hybrid search does most of the work: keyword matching catches exact part codes and parameter "
        "names that semantic search blurs. Reranking adds a smaller ranking gain."
    )

    with st.expander("All metrics as a table"):
        metrics = results["metrics"]
        table = pd.DataFrame(
            {METRIC_NAMES[m]: [summary[mode].get(m) for mode in MODE_NAMES] for m in metrics},
            index=list(MODE_NAMES.values()),
        )
        table["Latency (s)"] = [summary[mode]["latency_s"] for mode in MODE_NAMES]
        st.dataframe(table.style.format("{:.3f}", na_rep="pending"), width="stretch")
        if not {"faithfulness", "accuracy"} <= set(metrics) or not results.get("complete", True):
            st.caption("The LLM-judged answer metrics are still being collected (free judge quota: 200K tokens/day).")

    st.subheader("Keyword vs paraphrased questions")
    style_rows = []
    for style, by_mode in results["by_style"].items():
        count = sum(1 for q in results["per_question"] if q["style"] == style)
        for mode, scores in by_mode.items():
            style_rows.append({"Questions": f"{style} ({count})", "Mode": MODE_NAMES[mode],
                               "Hit@5": scores["hit"], "MRR": scores["mrr"]})
    st.dataframe(
        pd.DataFrame(style_rows),
        hide_index=True,
        width="stretch",
        column_config={"Hit@5": st.column_config.NumberColumn(format="percent"),
                       "MRR": st.column_config.NumberColumn(format="%.3f")},
    )

    st.subheader("Per question")
    st.caption("Position of the first answer-bearing chunk in each mode's top 5 (miss = not retrieved).")
    rows = []
    for q in results["per_question"]:
        row = {"ID": q["id"], "Style": q["style"], "Question": q["question"]}
        for mode, name in MODE_NAMES.items():
            rank = q["first_rank"][mode]
            row[name] = f"#{rank}" if rank else "miss"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=420)

    reranker_runs = sorted(RERANKERS_DIR.glob("*.json")) if RERANKERS_DIR.exists() else []
    if reranker_runs:
        st.subheader("Smaller rerankers")
        st.caption("The same test set, swapping only the cross-encoder (retrieval metrics, no LLM).")
        rows = []
        for path in reranker_runs:
            run = load_json(path)
            scores = run["summary"]["hybrid+rerank"]
            rows.append({"Reranker": run["rerank_model"], "Hit@5": scores["hit"], "MRR": scores["mrr"],
                         "Context recall": scores["recall"], "Latency (s)": scores["latency_s"]})
        st.dataframe(
            pd.DataFrame(rows).sort_values("MRR", ascending=False),
            hide_index=True,
            width="stretch",
            column_config={"Hit@5": st.column_config.NumberColumn(format="percent"),
                           "MRR": st.column_config.NumberColumn(format="%.3f"),
                           "Context recall": st.column_config.NumberColumn(format="%.3f"),
                           "Latency (s)": st.column_config.NumberColumn(format="%.2f")},
        )
