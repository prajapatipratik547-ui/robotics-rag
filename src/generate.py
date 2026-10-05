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

import groq
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
- If the context has nothing that answers the question, reply with exactly this sentence and nothing else: "{NOT_FOUND}"
- If the context answers only part of the question, answer that part and say in one short sentence which part the documents don't cover. Never add the sentence above to an answer.
- Be concise: a few sentences or a short list."""

RETRYABLE_STATUS = {429, 500, 503}  # rate limit / transient server errors
MAX_ATTEMPTS = 4
MAX_RETRY_WAIT_S = 60  # a 429 asking us to wait longer is a daily quota, not a blip


class QuotaExceeded(RuntimeError):
    """The provider's rate limit is still hit after retrying (usually a daily cap)."""


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
def get_groq() -> groq.Groq:
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to the .env file in the project root.")
    # The SDK itself retries 429s and 5xx errors, waiting as long as Groq's
    # retry-after header says (per-minute token limits reset within seconds).
    return groq.Groq(api_key=config.GROQ_API_KEY, max_retries=5)


@lru_cache(maxsize=1)
def get_gemini() -> genai.Client:
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to the .env file in the project root.")
    return genai.Client(api_key=config.GEMINI_API_KEY)


def _retry_delay_s(error: errors.APIError) -> float | None:
    """The server's suggested wait from a 429's RetryInfo, e.g. '29119s'."""
    try:
        for detail in error.details["error"]["details"]:
            if detail.get("@type", "").endswith("RetryInfo"):
                return float(detail["retryDelay"].rstrip("s"))
    except (KeyError, TypeError, ValueError, AttributeError):
        pass
    return None


def call_llm(system: str, prompt: str) -> str:
    """Send one system + user prompt to the configured provider, return its text.
    The provider-specific code lives only here and in the two functions below."""
    if config.LLM_PROVIDER == "groq":
        return _call_groq(system, prompt)
    if config.LLM_PROVIDER == "gemini":
        return _call_gemini(system, prompt)
    raise ValueError(f"Unknown LLM_PROVIDER {config.LLM_PROVIDER!r}; use 'groq' or 'gemini'.")


def _call_groq(system: str, prompt: str) -> str:
    try:
        response = get_groq().chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0,  # same question -> same answer, which keeps eval stable
            reasoning_effort="none",  # answer directly; grounded lookup needs no thinking
            max_completion_tokens=config.ANSWER_MAX_TOKENS,
        )
    except groq.RateLimitError as e:
        # Still limited after the SDK's retries: a daily limit, not a blip.
        raise QuotaExceeded(f"Groq rate limit for {config.GROQ_MODEL}: {e.message}") from e
    return (response.choices[0].message.content or "").strip()


def _call_gemini(system: str, prompt: str) -> str:
    """Retries per-minute rate limits and transient server errors with backoff
    (2 s, 4 s, 8 s, or the server's suggested wait). Fails fast with a clear
    message when the daily quota is used up, since retrying for hours is
    pointless."""
    llm_config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0,  # same question -> same answer, which keeps eval stable
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = get_gemini().models.generate_content(
                model=config.GEMINI_MODEL, contents=prompt, config=llm_config
            )
            return (response.text or "").strip()
        except errors.APIError as e:
            if e.code not in RETRYABLE_STATUS or attempt == MAX_ATTEMPTS:
                raise
            suggested = _retry_delay_s(e) if e.code == 429 else None
            if suggested is not None and suggested > MAX_RETRY_WAIT_S:
                raise QuotaExceeded(
                    f"Gemini free-tier quota for {config.GEMINI_MODEL} is used up; it resets in "
                    f"~{suggested / 3600:.1f} h. Check usage at https://ai.dev/rate-limit"
                ) from e
            time.sleep(max(2**attempt, suggested or 0))
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
