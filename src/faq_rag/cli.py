"""Command-line interface for the naive RAG pipeline."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import httpx
import typer
from dotenv import load_dotenv

from faq_rag.benchmark import DEFAULT_MODELS, run_benchmark
from faq_rag.chunker import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP_CHARS
from faq_rag.crawler import filter_article_urls, load_sitemap_urls, run_crawl
from faq_rag.evaluator import run_evaluation
from faq_rag.generator import default_model, stream_answer
from faq_rag.indexer import build_index
from faq_rag.retriever import Retriever, RetrievedChunk

load_dotenv()

app = typer.Typer(help="Naive RAG over the Dropbox Help Center.")


@app.callback()
def main() -> None:
    """Naive RAG over the Dropbox Help Center: crawl, index, ask, chat, eval."""


@app.command()
def crawl(
    limit: int | None = typer.Option(None, help="Only consider the first N candidate article URLs."),
    concurrency: int = typer.Option(3, help="Max concurrent requests."),
    delay: float = typer.Option(0.5, help="Polite delay (seconds) before each request starts."),
    data_dir: Path = typer.Option(Path("data"), help="Directory for the sitemap cache and output."),
) -> None:
    """Crawl the sitemap and save parsed English articles to data/articles.jsonl."""
    data_dir.mkdir(parents=True, exist_ok=True)
    sitemap_cache = data_dir / "sitemap.xml"
    articles_path = data_dir / "articles.jsonl"

    with httpx.Client(timeout=30.0) as client:
        all_urls = load_sitemap_urls(sitemap_cache, client)
    candidate_urls = filter_article_urls(all_urls)
    print(f"Sitemap: {len(all_urls)} URLs total, {len(candidate_urls)} English articles (>=2 path segments).")

    if limit is not None:
        candidate_urls = candidate_urls[:limit]
        print(f"Limiting to the first {limit} candidates for this run.")

    summary = asyncio.run(run_crawl(candidate_urls, articles_path, concurrency, delay))
    print(f"\nDone. saved={summary['saved']} skipped={summary['skipped']} failed={summary['failed']}")
    if summary["aborted_early"]:
        print("(stopped early due to sustained rate-limiting - re-run to continue)")


@app.command()
def index(
    data_dir: Path = typer.Option(Path("data"), help="Directory containing articles.jsonl."),
    max_chars: int = typer.Option(DEFAULT_MAX_CHARS, help="Target chunk size, in characters."),
    overlap_chars: int = typer.Option(DEFAULT_OVERLAP_CHARS, help="Overlap between size-split chunks."),
) -> None:
    """Chunk and embed data/articles.jsonl into the local Weaviate index (data/weaviate/)."""
    articles_path = data_dir / "articles.jsonl"
    count = build_index(articles_path, data_dir, max_chars, overlap_chars)
    print(f"\nDone. Indexed {count} chunks -> {data_dir / 'weaviate'}/")


def _status(message: str) -> None:
    """Progress notes go to stderr, so stdout is just the answer and its sources."""
    typer.echo(message, err=True)


def _print_stream(pieces: Iterator[str]) -> str:
    """Print the answer as the model writes it, and return the full text."""
    parts = []
    for piece in pieces:
        print(piece, end="", flush=True)
        parts.append(piece)
    print()
    return "".join(parts)


def _print_sources(chunks: list[RetrievedChunk]) -> None:
    print("Sources:")
    for i, chunk in enumerate(chunks, start=1):
        print(f"  [{i}] {chunk.title} - {chunk.url}  (score={chunk.score:.3f})")


def _print_llm_error(exc: Exception) -> None:
    typer.echo(
        f"Error calling the LLM: {exc}\n"
        "By default this expects a local Ollama server at http://localhost:11434 - start it "
        f"with `ollama serve` and run `ollama pull {default_model()}`, or set "
        "OPENAI_BASE_URL/LLM_MODEL to point elsewhere.",
        err=True,
    )


@app.command()
def ask(
    query: str,
    top_k: int = typer.Option(5, help="Number of chunks to retrieve."),
    retrieve_only: bool = typer.Option(
        False, "--retrieve-only", help="Skip the LLM call; just show retrieved chunks."
    ),
    data_dir: Path = typer.Option(Path("data"), help="Directory containing the index."),
) -> None:
    """Answer a question using retrieval-augmented generation over the crawled articles."""
    _status("Searching the help center...")
    with Retriever(data_dir) as retriever:
        chunks = retriever.search(query, top_k=top_k)

    if not retrieve_only:
        _status(
            f"Answering with {default_model()} - on a CPU it reads the excerpts for "
            "~30-60 s before the first words appear...\n"
        )
        try:
            _print_stream(stream_answer(query, chunks))
        except Exception as exc:
            _print_llm_error(exc)
            raise typer.Exit(1) from exc
        print()

    _print_sources(chunks)


@app.command()
def chat(
    top_k: int = typer.Option(5, help="Number of chunks to retrieve per turn."),
    data_dir: Path = typer.Option(Path("data"), help="Directory containing the index."),
) -> None:
    """Interactive chat with conversation memory: the LLM sees prior turns,
    and retrieval also uses the previous question, so follow-ups like
    "and on the Family plan?" still find the right article."""
    history: list[dict[str, str]] = []
    previous_query = ""
    _status("Loading the search index...")

    with Retriever(data_dir) as retriever:
        print("Ask a question about Dropbox (type 'exit' or press Ctrl+D to quit).")
        print(f"Answers come from {default_model()} on your CPU: expect ~30-60 s before each one starts.\n")
        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query or query.lower() in {"exit", "quit"}:
                break

            # Naive follow-up handling - plain concatenation, no LLM query
            # rewriting. Trade-off: an abrupt topic change drags the old topic
            # into this turn's search for one turn.
            chunks = retriever.search(f"{previous_query} {query}".strip(), top_k=top_k)
            previous_query = query
            print("\nAssistant: ", end="", flush=True)
            try:
                answer = _print_stream(stream_answer(query, chunks, history=history))
            except Exception as exc:
                print()
                _print_llm_error(exc)
                continue
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": answer})

            print()
            _print_sources(chunks)
            print()


@app.command(name="eval")
def eval_command(
    data_dir: Path = typer.Option(Path("data"), help="Directory containing the index."),
    questions_path: Path = typer.Option(Path("eval/questions.jsonl"), help="Path to the eval questions."),
) -> None:
    """Report retrieval metrics (hit@1/3/5, MRR) against eval/questions.jsonl."""
    run_evaluation(data_dir, questions_path)


@app.command()
def bench(
    models: str = typer.Option(",".join(DEFAULT_MODELS), help="Comma-separated Ollama model tags to compare."),
    limit: int | None = typer.Option(
        None, help="Only the first N questions - a quick smoke test (doesn't write the report)."
    ),
    data_dir: Path = typer.Option(Path("data"), help="Directory containing articles.jsonl and the index."),
    questions_path: Path = typer.Option(Path("eval/questions.jsonl"), help="Eval questions with key facts."),
    contexts_path: Path = typer.Option(Path("eval/contexts.jsonl"), help="Frozen top-5 contexts (created if missing)."),
    results_path: Path = typer.Option(Path("eval/results.md"), help="Where to write the report."),
) -> None:
    """Compare local LLMs on identical frozen contexts, scored by automatic checks."""
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    try:
        run_benchmark(model_list, questions_path, contexts_path, data_dir, results_path, limit)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
