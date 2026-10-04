"""Generation: turn retrieved chunks + a question into a cited answer.

The LLM sees only the numbered chunks and is told to answer from them
alone, citing each fact as [n]. We then map those numbers back to the
chunks' source files, so every citation points at a real retrieved chunk.
"""

import re
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai
from google.genai import errors, types

import config
from src.retrieve import Hit

NOT_FOUND = "I couldn't find the answer in the provided documents."

SYSTEM_PROMPT = f"""You answer questions about robotics hardware and software.

Rules:
- Use ONLY the numbered context passages provided. Do not use outside knowledge.
- After every sentence that states a fact, cite the passage(s) it came from as [1], [2], etc.
- Quote numbers, units, part codes, and pin or register names exactly as written in the context.
- If the context does not contain the answer, reply exactly: "{NOT_FOUND}"
- Be concise: a few sentences or a short list."""

RETRYABLE_STATUS = {429, 500, 503}  # rate limit / transient server errors
MAX_ATTEMPTS = 4


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]  # everything that was retrieved and shown to the LLM
    cited: list[tuple[int, Hit]]  # (number, chunk) for each [n] the LLM used


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = [f"[{i}] (source: {hit.citation})\n{hit.text}" for i, hit in enumerate(hits, start=1)]
    context = "\n\n".join(blocks)
    return f"Context passages:\n\n{context}\n\nQuestion: {question}"


@lru_cache(maxsize=1)
def get_llm() -> genai.Client:
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to the .env file in the project root.")
    return genai.Client(api_key=config.GEMINI_API_KEY)


def call_llm(system: str, prompt: str) -> str:
    """The only provider-specific function. Retries rate limits and transient
    server errors with exponential backoff (2 s, 4 s, 8 s)."""
    llm_config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0,  # same question -> same answer, which keeps eval stable
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = get_llm().models.generate_content(
                model=config.GEMINI_MODEL, contents=prompt, config=llm_config
            )
            return (response.text or "").strip()
        except errors.APIError as e:
            if e.code not in RETRYABLE_STATUS or attempt == MAX_ATTEMPTS:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def extract_citations(text: str, hits: list[Hit]) -> list[tuple[int, Hit]]:
    """Find every [n] (or [n, m]) in the answer and map it to its chunk.
    Numbers that don't match a retrieved chunk are ignored."""
    numbers: list[int] = []
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text):
        for n in group.split(","):
            n = int(n)
            if 1 <= n <= len(hits) and n not in numbers:
                numbers.append(n)
    return [(n, hits[n - 1]) for n in sorted(numbers)]


def generate(question: str, hits: list[Hit]) -> Answer:
    if not hits:
        return Answer(question, NOT_FOUND, hits, [])
    text = call_llm(SYSTEM_PROMPT, build_prompt(question, hits))
    return Answer(question, text, hits, extract_citations(text, hits))
