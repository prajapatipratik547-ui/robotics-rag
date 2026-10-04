# BUILD_SPEC.md — Robotics Hybrid RAG System

> This file is the single source of truth for building this project.
> It is written to be executed by Claude Code, stage by stage.
> Read it fully before starting. Build in the stage order given.
> Do not skip ahead. After each stage, follow the "How to work" rules below.

---

## 1. What we are building (plain English)

A question-answering system that answers questions about robotics hardware and
software by **looking up the answer in real documents** instead of guessing.

The user asks something like *"What is the scan range of the RPLiDAR A1M8?"* and
the system:
1. searches a set of robotics documents for the most relevant passages,
2. hands those passages to an LLM,
3. returns an answer grounded in those passages, **with a citation to the source**.

No hallucinated answers. Every answer is backed by a retrieved document chunk.

## 2. Why this specific design (the differentiator)

This is a **hybrid** RAG system. Retrieval combines:
- **dense** (semantic) search — captures meaning, and
- **sparse** (keyword / BM25) search — captures exact technical tokens like
  part numbers ("A1M8", "MPU6050") that semantic search misses,

then **fuses** the two result lists and **reranks** the top candidates with a
cross-encoder. This matters because the corpus is full of jargon and part codes
where keyword-only or meaning-only search each fail on their own.

The project is only "done" when retrieval quality is **measured** (see Stage 5).
The headline result is a before/after accuracy number.

## 3. Success criteria

- A user can ask a natural-language question and get a correct, cited answer.
- Hybrid + rerank measurably beats dense-only retrieval, proven with an eval.
- Clean public GitHub repo with a README a recruiter can understand in 2 minutes.
- Résumé outcome: *"Improved retrieval accuracy from X% to Y% by adding hybrid
  search and reranking, measured with a ragas evaluation harness."*

## 4. Hard constraints

- **Language:** Python 3.10+.
- **No Docker.** Run Qdrant in **local mode** via the Python client:
  `QdrantClient(path="./qdrant_data")`. (If a limitation is ever hit, the fallback
  is Qdrant Cloud free tier — a one-line client change. Do not introduce Docker.)
- **Free / local only.** Embeddings and reranker run locally on CPU. The only
  external call is the final LLM answer step, using a free tier.
- **Keep secrets in `.env`.** Never commit `.env`. Never hardcode API keys.
- **Explainability over cleverness.** Write the retrieval and fusion logic
  directly with the Qdrant client. Do NOT hide retrieval inside a high-level
  framework (no LangChain/LlamaIndex for the core pipeline). The user must be
  able to explain every step.
- **Small, readable modules.** Prefer clear functions over clever abstractions.

## 5. Tech stack (pinned choices)

- Vector store: **Qdrant** via `qdrant-client[fastembed]`, local mode.
- Dense embeddings: **BAAI/bge-small-en-v1.5** (via FastEmbed, CPU).
- Sparse retrieval: **BM25** (via FastEmbed, e.g. `Qdrant/bm25`).
- Fusion: Qdrant **Query API** with `prefetch` + **Reciprocal Rank Fusion (RRF)**.
- Reranker: **BAAI/bge-reranker-base** (cross-encoder, local, CPU).
- LLM (answer generation): **Groq** or **Google Gemini** free tier, key in `.env`.
- Evaluation: **ragas**.
- UI: **Streamlit** (added in Stage 6).
- Config: `.env` + a small `config.py`.

## 6. Repository structure (target)

```
robotics-rag/
├── BUILD_SPEC.md          # this file
├── README.md
├── requirements.txt
├── .gitignore             # ignores .venv, .env, qdrant_data/, __pycache__/
├── .env.example           # template (no real keys)
├── config.py              # model names, paths, constants
├── data/
│   ├── raw/               # source documents the user drops in (PDF/MD/HTML)
│   └── processed/         # cleaned/chunked output (optional cache)
├── src/
│   ├── ingest.py          # load → chunk → embed → store
│   ├── retrieve.py        # dense, sparse, hybrid, rerank
│   ├── generate.py        # build prompt from chunks → LLM → cited answer
│   └── rag.py             # ties retrieve + generate into one ask(question)
├── eval/
│   ├── testset.jsonl      # hand-built question/ground-truth set
│   └── run_eval.py        # runs ragas, compares retrieval modes
├── tests/
│   └── test_qdrant.py     # Stage 0 smoke test
└── app.py                 # Streamlit UI (Stage 6)
```

## 7. The corpus (user-supplied)

The user will place source documents in `data/raw/`. Expected contents:
- ROS2 / Nav2 documentation pages (Markdown or HTML or PDF).
- Component datasheets: RPLiDAR A1M8, Raspberry Pi 4B, Arduino Mega 2560,
  MPU6050, NEO-6M GPS, L298N, HC-SR04, MG996R.

If `data/raw/` is empty when a stage needs it, STOP and ask the user to add
documents, or add 2–3 small placeholder `.md` files so the pipeline can be
tested end to end, and clearly tell the user these are placeholders.

---

## 8. Build plan — stages

Build in this order. Each stage lists its objective, tasks, and a
**Definition of Done (DoD)** that must pass before moving on.

### Stage 0 — Setup
**Objective:** working environment + proof Qdrant runs locally.
- Create venv, `requirements.txt` (start with `qdrant-client[fastembed]`,
  `python-dotenv`), install.
- Create the folder structure above, `.gitignore`, `.env.example`, `config.py`.
- Write `tests/test_qdrant.py`: create a local-mode collection, insert 3 text
  points, run a similarity search, print results.
- `git init`, first commit.
**DoD:** `python tests/test_qdrant.py` prints the correct matching text.

### Stage 1 — Ingestion
**Objective:** turn `data/raw/` documents into searchable chunks in Qdrant.
- `src/ingest.py`: load PDF/MD/HTML, extract text, split into overlapping
  chunks (~500–800 tokens, ~100 overlap), attach metadata
  (`source_file`, `section` if available, `chunk_id`).
- Create a Qdrant collection with BOTH a dense vector and a sparse vector
  configured on each point (named vectors). Embed and upsert all chunks.
**DoD:** running ingestion reports N chunks stored; a quick count query confirms
the collection is populated with both dense and sparse vectors.

### Stage 2 — Baseline RAG (dense only)
**Objective:** first end-to-end answer, dense retrieval only.
- `src/retrieve.py`: `dense_search(query, k)` returning top-k chunks.
- `src/generate.py`: build a prompt that includes retrieved chunks + the
  question, instruct the LLM to answer ONLY from the context and cite the
  `source_file`. Call the free-tier LLM (key from `.env`).
- `src/rag.py`: `ask(question)` = dense_search → generate. Add a CLI entry so
  the user can ask questions from the terminal.
**DoD:** asking a question about a known document returns a correct, cited answer.
This is the BASELINE to beat.

### Stage 3 — Hybrid retrieval
**Objective:** add sparse search and fuse with dense.
- Add `sparse_search(query, k)` and `hybrid_search(query, k)` to `retrieve.py`.
- Implement hybrid using Qdrant's Query API: a `prefetch` for the dense vector
  and one for the sparse vector, combined with `FusionQuery` using RRF.
- Make the retrieval mode selectable (dense | sparse | hybrid) so modes can be
  compared later.
**DoD:** `hybrid_search` returns results; a query containing an exact part code
(e.g. "A1M8") retrieves the right chunk where dense-only was weaker.

### Stage 4 — Reranking
**Objective:** reorder the fused candidates for precision.
- Retrieve a larger candidate set (e.g. top 20) via hybrid, then rerank with
  the `bge-reranker-base` cross-encoder, keep top k (e.g. 5) for generation.
- Make reranking toggleable (on/off) for comparison.
**DoD:** with reranking on, the top chunks are visibly more relevant for a few
sample queries than without it.

### Stage 5 — Evaluation (the headline result)
**Objective:** measure retrieval quality and prove the design works.
- Build `eval/testset.jsonl`: 20–30 questions with the expected answer and/or
  the correct source chunk. (Ask the user to help with domain-correct answers.)
- `eval/run_eval.py`: run the test set through THREE modes — dense, hybrid,
  hybrid+rerank — and compute metrics with ragas (e.g. context precision /
  recall, answer faithfulness).
- Produce a small results table (print + save to `eval/results.md`).
**DoD:** a table showing dense vs hybrid vs hybrid+rerank, with hybrid+rerank
best. This yields the résumé number (X% → Y%).

### Stage 6 — UI + README + polish
**Objective:** make it presentable and shareable.
- `app.py`: Streamlit UI — text box for the question, shows the answer plus the
  retrieved sources/citations.
- Rewrite `README.md`: one-paragraph plain-English description, architecture
  diagram (describe the pipeline), tech stack, how to run, and the eval results
  table from Stage 5. Include a short "what I learned / design decisions" section.
**DoD:** `streamlit run app.py` works; README clearly explains the project and
shows the numbers.

---

## 9. How to work (rules for Claude Code)

1. **One stage at a time.** Complete a stage, meet its DoD, then STOP.
2. **After each stage, report to the user:** what you built, the key files, how
   you verified the DoD, and any decisions you made. Then WAIT for the user to
   say "go" before starting the next stage.
3. **Explain your decisions briefly** as you go (why this chunk size, what RRF
   does) — the user wants to understand the system, not just receive it.
4. **Commit at the end of every stage** with a clear message
   (e.g. `Stage 3: add hybrid retrieval with RRF fusion`). Keep the README's
   current-stage note updated.
5. **If something is ambiguous or a dependency/doc is missing, ASK** rather than
   guessing or inventing data. Never fabricate eval answers — get domain-correct
   ones from the user.
6. **Keep it runnable at every stage.** Never leave the repo in a broken state.
