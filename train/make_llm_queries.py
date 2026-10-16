"""Step 1, second attempt: synthetic training questions written by an LLM.

The first attempt used doc2query (make_queries.py), whose short web-search
style queries ("what is arduino mega 2560") made the fine-tuned model worse on
the real technical test questions. Here a free Groq model (gpt-oss-20b, whose
daily quota nothing else in the project uses) writes three specific questions
per chunk: one naming the exact part or parameter, two describing it in other
words. Same output format as make_queries.py, so filter_queries.py and
finetune.py take it unchanged.

Groq's free tier allows ~200K tokens/day for this model, about 750 chunks, so a
full run takes two days. Resumable: rerun it and it continues.

    python train/make_llm_queries.py
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import groq

import config
from src.passages import embedding_input
from src.testset import CHUNKS_PATH, load_jsonl

QUERIES_PATH = ROOT / "train" / "data" / "queries_llm.jsonl"
MODEL = "openai/gpt-oss-20b"
CHUNKS_PER_REQUEST = 6  # amortizes the instructions; output stays well under the token cap
SYSTEM = (
    "You write evaluation questions for a retrieval system over robotics datasheets and ROS 2 Nav2 docs. "
    "For each passage, write 3 questions a robotics engineer would actually ask that the passage answers on its own: "
    "one that names the exact part code or parameter, and two that describe the thing in other words without naming it. "
    "Be specific (ask for a value, default, limit, behaviour or reason), not generic like 'what is X'. "
    "If a passage holds nothing a question could ask about (e.g. only a table of contents), return no questions for it. "
    'Reply with JSON only: {"passages": [{"id": <passage number>, "questions": [strings]}]}'
)


def ask(client: groq.Groq, batch: list[dict]) -> dict[int, list[str]]:
    user = "\n\n".join(f"Passage {i}:\n{embedding_input(c)}" for i, c in enumerate(batch, start=1))
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        temperature=0.7,
        max_completion_tokens=3000,
        reasoning_effort="low",
        response_format={"type": "json_object"},
    )
    passages = json.loads(response.choices[0].message.content)["passages"]
    return {int(p["id"]): [q.strip() for q in p["questions"] if isinstance(q, str) and q.strip()] for p in passages}


def main() -> None:
    chunks = load_jsonl(CHUNKS_PATH)
    QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = {row["chunk_id"] for row in load_jsonl(QUERIES_PATH)} if QUERIES_PATH.exists() else set()
    # A chunk the model gave no questions for is recorded too (query None), so it isn't asked again.
    todo = [c for c in chunks if c["chunk_id"] not in done]
    print(f"{len(chunks)} chunks, {len(done)} done, {len(todo)} to go")
    client = groq.Groq(api_key=config.GROQ_API_KEY, max_retries=6)  # retries per-minute limits itself

    with QUERIES_PATH.open("a", encoding="utf-8") as out:
        for start in range(0, len(todo), CHUNKS_PER_REQUEST):
            batch = todo[start : start + CHUNKS_PER_REQUEST]
            try:
                questions = ask(client, batch)
            except groq.RateLimitError as e:
                print(f"Daily limit reached; rerun later to continue. ({e.message[:120]})")
                sys.exit(2)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, groq.APIError) as e:
                print(f"Skipping a batch for now ({type(e).__name__}); a rerun retries it.")
                continue
            for i, chunk in enumerate(batch, start=1):
                own = questions.get(i, [])
                for query in dict.fromkeys(own) or [None]:
                    out.write(json.dumps({"chunk_id": chunk["chunk_id"], "query": query}, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{start + len(batch)}/{len(todo)} chunks", flush=True)
            time.sleep(1)


if __name__ == "__main__":
    main()
