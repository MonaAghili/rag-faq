"""Build the RAG prompt and call a local LLM through Ollama.

Defaults to Ollama's OpenAI-compatible endpoint (http://localhost:11434/v1) -
free, local, no API key. Override OPENAI_BASE_URL/LLM_MODEL to point at any
other OpenAI-compatible server instead.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import dataclass

import openai

from faq_rag.retriever import RetrievedChunk

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "granite4:3b"
MAX_TOKENS = 512
# Keeps chat under the 4096-token context the Ollama server is started with:
# ~2.2k tokens of excerpts + system prompt + 3 prior exchanges + 512 to answer.
# Ollama truncates overflow from the *start* of the prompt - i.e. it would
# silently drop the system instructions first.
MAX_HISTORY_MESSAGES = 6

SYSTEM_PROMPT = """You are a support assistant for the Dropbox Help Center.

Answer the user's question using ONLY the numbered excerpts provided below \
- do not use any outside knowledge, even if you know the answer.

If the excerpts don't contain enough information to answer, say plainly that \
you don't know and suggest the user check Dropbox support directly. Do not guess.

When the answer involves a procedure, give numbered steps.

Cite your sources inline: right after each fact, put the number of the excerpt \
it came from in square brackets, for example [2]."""


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    seconds: float


def build_context_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    excerpts = "\n\n".join(f"[{i}] {c.title} ({c.url})\n{c.text}" for i, c in enumerate(chunks, start=1))
    return f"Excerpts:\n\n{excerpts}\n\nQuestion: {query}"


def build_messages(
    query: str,
    chunks: list[RetrievedChunk],
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """`history` is a plain transcript of prior {"role", "content"} turns (no
    excerpts in it), so a follow-up like "what about on mobile?" has context.
    Only the current turn is stuffed with excerpts: retrieval stays
    independent per turn (no query rewriting in a naive RAG pipeline)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *(history or [])[-MAX_HISTORY_MESSAGES:],
        {"role": "user", "content": build_context_prompt(query, chunks)},
    ]


def default_model() -> str:
    return os.environ.get("LLM_MODEL", DEFAULT_MODEL)


def base_url() -> str:
    return os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)


def _client() -> openai.OpenAI:
    return openai.OpenAI(base_url=base_url(), api_key=os.environ.get("OPENAI_API_KEY", "ollama"))


def complete(
    messages: list[dict[str, str]],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
) -> Completion:
    """One chat completion, plus the token counts and wall time the benchmark needs."""
    start = time.perf_counter()
    response = _client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    seconds = time.perf_counter() - start

    choice = response.choices[0]
    usage = response.usage
    return Completion(
        text=choice.message.content or "",
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        finish_reason=choice.finish_reason or "",
        seconds=seconds,
    )


def stream_answer(
    query: str,
    chunks: list[RetrievedChunk],
    history: list[dict[str, str]] | None = None,
) -> Iterator[str]:
    """Yield the answer piece by piece as the default model writes it, so
    `ask`/`chat` show text immediately instead of after the whole answer.
    Same prompt and settings the benchmark measured."""
    stream = _client().chat.completions.create(
        model=default_model(),
        messages=build_messages(query, chunks, history),
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        stream=True,
    )
    for event in stream:
        if event.choices and event.choices[0].delta.content:
            yield event.choices[0].delta.content
