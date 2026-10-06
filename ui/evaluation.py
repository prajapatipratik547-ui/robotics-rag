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


def bar_chart(summary: dict, metric: str, cis: dict | None) -> alt.Chart:
    """One bar per mode, with its 95% confidence interval as a whisker when known."""
    rows = []
    for mode, name in MODE_NAMES.items():
        score = summary[mode].get(metric)
        if score is None:
            continue
        low, high = cis["modes"][mode][metric] if cis else (score, score)
        rows.append({"Mode": name, "Score": score, "Low": low, "High": high})
    data = pd.DataFrame(rows)
    is_percent = metric == "hit"
    fmt = ".0%" if is_percent else ".3f"  # value labels and tooltips
    tick_fmt = ".0%" if is_percent else ".1f"
    # Quarter ticks read well as percentages; MRR gets 0.2 steps so ".1f" never rounds a tick.
    ticks = [0, 0.25, 0.5, 0.75, 1] if is_percent else [0, 0.2, 0.4, 0.6, 0.8, 1]
    y =alt.Y("Mode:N", sort=list(MODE_NAMES.values()), title=None, axis=alt.Axis(labelLimit=200))
    x_scale = alt.Scale(domain=[0, 1.12])
    tooltip = [
        alt.Tooltip("Mode:N"),
        alt.Tooltip("Score:Q", format=fmt, title=METRIC_NAMES[metric]),
        alt.Tooltip("Low:Q", format=fmt, title="95% CI low"),
        alt.Tooltip("High:Q", format=fmt, title="95% CI high"),
    ]
    chart = alt.Chart(data, title=METRIC_NAMES[metric])
    bars = chart.mark_bar(cornerRadiusEnd=4, size=22).encode(
        y=y,
        x=alt.X("Score:Q", scale=x_scale, title=None, axis=alt.Axis(format=tick_fmt, values=ticks)),
        color=alt.Color("Mode:N", scale=alt.Scale(domain=list(MODE_COLORS), range=list(MODE_COLORS.values())), legend=None),
        tooltip=tooltip,
    )
    whiskers = chart.mark_rule(color="#e6e9f0", strokeWidth=2, opacity=0.7).encode(
        y=y, x=alt.X("Low:Q", scale=x_scale), x2="High:Q", tooltip=tooltip
    )
    # The label sits past the whisker so the two never overlap.
    labels = chart.mark_text(align="left", dx=6, color="#e6e9f0", fontSize=13).encode(
        y=y, x=alt.X("High:Q", scale=x_scale), text=alt.Text("Score:Q", format=fmt)
    )
    return (
        (bars + whiskers + labels)
        .properties(height=alt.Step(40))  # one 40px band per bar: 22px bar + visible gap
        .configure(background="#151b2b")
        .configure_view(stroke=None)
        .configure_axis(labelColor="#b4bccb", gridColor="rgba(255,255,255,0.07)", domainColor="#383835", tickColor="#383835")
        .configure_title(color="#e6e9f0", anchor="start", fontSize=14)
    )


def show_differences(cis: dict) -> None:
    """The paired mode differences, and whether each is distinguishable from noise."""
    st.subheader("Is the difference real?")
    rows = []
    for row in cis["differences"]:
        a, b = (MODE_NAMES[m] for m in row["modes"])
        hit, mrr = row["hit"], row["mrr"]
        clear = any(r["ci"][0] > 0 or r["ci"][1] < 0 for r in (hit, mrr))
        rows.append({
            "Comparison": f"{a} vs {b}",
            "Hit@5": f"{hit['diff'] * 100:+.0f} pts ({hit['ci'][0] * 100:+.0f} to {hit['ci'][1] * 100:+.0f})",
            "MRR": f"{mrr['diff']:+.3f} ({mrr['ci'][0]:+.3f} to {mrr['ci'][1]:+.3f})",
            "Verdict": "✓ clear gain" if clear else "within noise",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Paired bootstrap over the test questions (10,000 resamples); the 95% interval is in brackets. "
               "A clear gain's interval excludes 0.")


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

    cis = results.get("confidence_intervals")
    left, right = st.columns(2)
    left.altair_chart(bar_chart(summary, "hit", cis), width="stretch")
    right.altair_chart(bar_chart(summary, "mrr", cis), width="stretch")
    st.caption(
        "Whiskers: 95% bootstrap confidence intervals. Hybrid search does the work: keyword matching "
        "catches exact part codes and parameter names that semantic search blurs. Reranking reorders the "
        "top 5 a little, but its gain is within the noise."
    )
    if cis:
        show_differences(cis)

    with st.expander("All metrics as a table"):
        metrics = results["metrics"]
        llm_modes = results.get("llm_modes", list(MODE_NAMES))
        table = pd.DataFrame(
            {METRIC_NAMES[m]: [summary[mode].get(m) for mode in MODE_NAMES] for m in metrics},
            index=list(MODE_NAMES.values()),
        )
        table["Latency (s)"] = [summary[mode]["latency_s"] for mode in MODE_NAMES]
        st.dataframe(table.style.format("{:.3f}", na_rep="–"), width="stretch")
        if not {"faithfulness", "accuracy"} <= set(metrics) or not results.get("complete", True):
            st.caption("The LLM-judged answer metrics are still being collected (free judge quota: 200K tokens/day).")
        else:
            st.caption(f"Answer metrics are measured for {', '.join(MODE_NAMES[m] for m in llm_modes)} only "
                       "(the free judge quota allows one mode).")

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
