# Robotics Hybrid RAG

Ask a question about robot hardware or navigation software, such as *"What is
the I2C address of the MPU-6050 when AD0 is high?"*, and get an answer looked
up in the real datasheets and ROS 2 Nav2 docs, with a citation to the exact
page or section it came from. If the documents don't contain the answer, it
says so instead of guessing. Retrieval combines **semantic search** (good at
meaning) with **keyword search** (good at exact part codes like `A1M8` and
parameter names like `inflation_radius`), then a **cross-encoder reranker**
puts the best passages first. Everything except the final answer-writing step
runs locally on a CPU, for free.

## Results

Measured on a hand-built test set of 26 questions, each with a confirmed
answer and the exact passage(s) in the documents that contain it
([eval/testset.jsonl](eval/testset.jsonl)). Full report:
[eval/results.md](eval/results.md).

| Retrieval mode | Hit@5 | MRR | Context recall | Context precision | Latency (CPU) |
|---|---|---|---|---|---|
| Dense only (baseline) | 88% | 0.815 | 0.721 | 0.262 | 0.01 s |
| Hybrid (dense + BM25, RRF) | **96%** | 0.891 | **0.801** | **0.292** | 0.02 s |
| Hybrid + rerank | **96%** | **0.905** | 0.798 | 0.285 | ~4.5 s |

- **Hit@5** is the retrieval accuracy: the share of questions where a chunk
  containing the answer is among the 5 passages handed to the LLM. If it isn't
  there, the LLM can't answer correctly.
- **MRR** (mean reciprocal rank) rewards putting that chunk first: 1.0 means it
  was always ranked #1.
- **Context recall / precision** are ragas' ID-based metrics: the share of the
  answer-bearing chunks that were retrieved, and the share of retrieved chunks
  that bear the answer.

**Headline:** adding hybrid search and reranking improved retrieval accuracy
from 88% to 96% and MRR from 0.815 to 0.905. Almost all of the accuracy gain
comes from hybrid search; reranking adds a smaller ranking gain at a real
latency cost (see design decisions below).

The answer-level ragas metrics (faithfulness to the retrieved passages, and
accuracy against the reference answer, both judged by a separate LLM) are
still being collected: the free judge model allows 200K tokens/day, about a
day and a half for the full run. They will be added to
[eval/results.md](eval/results.md) when complete.

## How it works

```mermaid
flowchart LR
    Q[Question] --> D[Dense search<br/>bge-small-en-v1.5]
    Q --> S[Keyword search<br/>BM25]
    D -- top 20 --> F[RRF fusion]
    S -- top 20 --> F
    F -- top 20 --> R[Cross-encoder rerank<br/>bge-reranker-base]
    R -- top 5 --> L[LLM writes the answer<br/>from those 5 only]
    L --> A[Answer with citations]
```

**Ingestion** (`src/ingest.py`, run once): load the PDFs, Markdown and HTML
in `data/raw/`, split them by page or heading, and cut them into chunks of up
to 400 tokens with 100 tokens of overlap. Each chunk is stored in a local
Qdrant database with two vectors: a dense embedding for meaning and a sparse
BM25 vector for keywords.

**Answering a question:**

1. **Dense search** embeds the question and finds the 20 chunks closest in
   meaning (cosine similarity).
2. **Keyword search** (BM25) finds the 20 chunks that best match the
   question's exact words, weighting rare words like part codes highest.
3. **Reciprocal Rank Fusion (RRF)** merges the two lists by rank: a chunk
   ranked high by both methods beats one ranked high by only one.
4. **Reranking:** a cross-encoder reads the question and each of the 20
   chunks together, scores how well the chunk answers it, and keeps the
   best 5.
5. **Generation:** the LLM gets only those 5 numbered passages, with
   instructions to answer from them alone and cite each fact as `[n]`. The
   code maps every `[n]` back to its source file and page or section.

Retrieval mode (dense / sparse / hybrid) and reranking can be switched in the
app sidebar or on the command line, which is how the modes were compared.

## Tech stack

| Part | Choice | Runs |
|---|---|---|
| Vector database | Qdrant, local mode (no server, no Docker) | local |
| Dense embeddings | BAAI/bge-small-en-v1.5 via FastEmbed (384-dim) | local CPU |
| Keyword search | BM25 (`Qdrant/bm25` via FastEmbed, IDF computed by Qdrant) | local CPU |
| Fusion | Qdrant Query API: `prefetch` + `FusionQuery(RRF)` | local |
| Reranker | BAAI/bge-reranker-base cross-encoder via FastEmbed | local CPU |
| Answer LLM | `qwen/qwen3.8-27b` on Groq's free tier (Gemini 2.5 Flash also supported) | API |
| Evaluation | ragas 0.4, with `openai/gpt-oss-120b` on Groq as the judge | API |
| UI | Streamlit | local |
| Parsing | PyMuPDF (PDF), Beautiful Soup (HTML) | local |

Retrieval and fusion are written directly against the Qdrant client in under
200 lines of [src/retrieve.py](src/retrieve.py), with no LangChain or
LlamaIndex, so every step can be read and explained.

## Run it

Needs Python 3.10+ (developed on 3.13, Windows) and a free Groq API key from
[console.groq.com](https://console.groq.com).

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env              # then paste your key after GROQ_API_KEY=
python data/download_corpus.py      # download the 32 source documents into data/raw/
python src/ingest.py                # chunk, embed and store them (a few minutes)
streamlit run app.py                # open the app in your browser
```

The first run downloads the three models (~1.2 GB, mostly the reranker) into
`.fastembed_cache/`.

Command-line alternatives:

```powershell
python src/rag.py "What is the stall torque of the MG996R?"   # one question
python src/rag.py --show-context                               # interactive, shows retrieved chunks
python src/rag.py --mode dense --no-rerank "Is MPPI supported?" # pick a retrieval mode
python src/retrieve.py "A1M8 scan frequency"   # compare all modes' top 5 side by side
python eval/run_eval.py --retrieval-only       # retrieval metrics, no LLM calls (~2 min)
python eval/run_eval.py                        # plus LLM-judged answer metrics (rate-limited, resumable)
python tests/test_qdrant.py                    # smoke test: local Qdrant works
```

Qdrant's local mode lets only one process open the database at a time, so
close the app before running the CLI or the eval, and the other way round.
The eval releases the database as soon as its retrieval step finishes.

To use Gemini instead of Groq, set `LLM_PROVIDER = "gemini"` in
[config.py](config.py) and `GEMINI_API_KEY` in `.env`. Its free tier allowed
only 20 requests/day on this project's key, too few for the eval.

## Corpus

32 documents, split into 906 chunks:

- **9 datasheets:** RPLiDAR A1M8, Raspberry Pi 4 Model B, Arduino Mega 2560,
  MPU-6050 (product specification and register map), NEO-6M GPS, L298N motor
  driver, HC-SR04 ultrasonic sensor, MG996R servo.
- **23 Nav2 documentation pages** (ROS 2 Jazzy): navigation concepts, robot
  setup guides, the controller, planner, behavior and BT navigator servers,
  costmap layers, AMCL, the RPP and MPPI controllers, the NavFn and Smac
  Hybrid-A* planners, the tuning guide, and the SLAM and GPS tutorials.

The documents are copyrighted by their publishers, so they are not stored in
this repository. [data/download_corpus.py](data/download_corpus.py) downloads
them and records where each one comes from. To add your own documents, drop
PDF, Markdown or HTML files into `data/raw/` and rerun `python src/ingest.py`.
Name files descriptively (e.g. `rplidar_a1m8_datasheet.pdf`): the file name
becomes the document title that is prepended to each chunk before embedding.

## Project structure

```
app.py                  Streamlit UI
config.py               every model name, path and setting, in one place
src/ingest.py           load -> chunk -> embed -> store
src/retrieve.py         dense, sparse and hybrid search, reranking
src/generate.py         prompt -> LLM -> answer with citations mapped to sources
src/rag.py              ask(question) = retrieve + generate; command-line entry point
eval/testset.jsonl      26 questions, reference answers, evidence phrases
eval/run_eval.py        compares dense, hybrid and hybrid+rerank; writes eval/results.md
data/download_corpus.py downloads the source documents
tests/test_qdrant.py    smoke test for local Qdrant
```

## Design decisions and what I learned

- **Keyword search matters for technical documents.** Embedding models blur
  exact tokens: a question naming `motion_model` or "Mega 2560" can land on a
  chunk about the right topic that doesn't hold the answer. BM25 catches those
  exact matches, which is where hybrid search won most of its questions
  (Hit@5 88% to 96%).
- **RRF instead of score blending.** Cosine similarities and BM25 scores live
  on different scales, so adding them needs tuned weights. RRF uses only each
  chunk's rank in each list, so it needs no tuning.
- **Reranking was a smaller win than expected.** It improved ranking (MRR
  0.891 to 0.905) but not Hit@5, and costs ~4.5 s per question on CPU against
  0.02 s for hybrid alone. Once hybrid search already finds the answer for
  96% of questions, there is little left for a reranker to fix. It also made
  some ranking mistakes, such as preferring a power-consumption table whose
  footnotes mention "cold start" over the table that holds the cold-start time. I kept it on by
  default for answer quality, with a toggle, and reported the result as
  measured rather than tuning settings until it won.
- **Chunks of 400 tokens, counted with the embedding model's own tokenizer.**
  Both models truncate input at 512 tokens, so 400 leaves room for the
  document title and section name prepended to each chunk, and for the
  question at rerank time. The prefix matters for parameter reference pages,
  where a parameter's name appears only in its heading.
- **Label the evidence, not chunk IDs.** Each test question names an exact
  phrase from the document that answers it; the eval finds every chunk
  containing that phrase. The test set survives re-chunking, and a chunk that
  repeats the answer elsewhere still counts.
- **Evaluate the evaluation.** An audit of all 26 questions found five whose
  answer also appeared in chunks I hadn't labelled, such as the MPU-6050
  register map repeating the gyro ranges from the product specification.
  Fixing those raised every mode's score. The audit covered all questions, not
  just those where a mode lost, so the labels weren't fitted to the results.
- **A different model judges the answers** from the one that writes them, so
  the judge isn't grading its own work. The answer metrics also caught a
  prompt bug: the model sometimes gave a correct answer and then added "I
  couldn't find the answer", which I fixed in the prompt.
- **Check free-tier limits yourself.** Third-party guides overstated Gemini's
  free quota (20 requests/day in practice, not ~1,500). Groq's limits were read
  from its rate-limit response headers, and its daily token cap from a real
  429 error. Rate limits shaped the eval design: deterministic retrieval
  metrics with no LLM, plus cached, resumable LLM-judged metrics.
- **Local Qdrant on Windows:** deleting a collection didn't remove its data
  file, so old chunks came back after re-ingestion. Ingestion now deletes the
  storage folder before rebuilding, and verifies that the stored point count
  matches the chunk count.

## Limitations

- The test set is small (26 questions) and was written from the corpus by the
  builder, so the numbers show relative differences between modes better than
  absolute real-world accuracy.
- PDF tables are extracted as flat text, so values in complex tables can be
  hard to retrieve and read (the NEO-6M time-to-first-fix table is one example).
- Reranking on CPU takes several seconds per question; a GPU, or reranking
  fewer candidates, would cut that.
