"""Benchmark local LLMs on identical, frozen RAG contexts.

`faq-rag bench` does, for the questions in eval/questions.jsonl:

1. Frozen contexts: each question's top-5 chunks are retrieved once and stored
   as chunk IDs in eval/contexts.jsonl (committed - but no Dropbox text in it).
   The text is rebuilt from data/articles.jsonl with chunk_article(), and a
   short hash per chunk catches drift, so every model - and every later run -
   answers from exactly the same excerpts.
2. Validation: every key fact must appear in its question's frozen excerpts,
   and a bait string may appear in neither the excerpts nor the question.
3. Generation through the production prompt (generator.build_messages +
   complete), three answers per question - one per prompt-cache state (see
   CACHE_STATES) - cached in data/bench/answers/<model>.jsonl under a hash of
   (model, messages, cache state): a crashed run resumes, and re-scoring after
   fixing a check in scoring.py never needs the answers regenerated.
4. Scoring pooled over the three answers, two free baselines, a bootstrap 95%
   CI, a winner, and a markdown report in eval/results.md.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from faq_rag.chunker import chunk_article
from faq_rag.evaluator import EvalQuestion, load_questions
from faq_rag.generator import MAX_TOKENS, SYSTEM_PROMPT, base_url, build_messages, complete
from faq_rag.indexer import load_articles
from faq_rag.retriever import RetrievedChunk, Retriever
from faq_rag.scoring import AnswerScore, contains_fact, light_normalize, normalize, score_answer

DEFAULT_MODELS = ["granite4:3b", "qwen2.5:3b", "phi4-mini:3.8b", "llama3.2:3b"]
CONTEXT_TOP_K = 5
NUM_CTX = 4096  # must match OLLAMA_CONTEXT_LENGTH the Ollama server was started with
CITATION_GATE = 0.9
DECLINE_GATE = 0.8  # with 6 unanswerable questions x 3 answers: at least 15 of 18 declined
LATENCY_NOISE = 0.25  # speed gaps smaller than this don't break a tie
BOOTSTRAP_RESAMPLES = 2000

# Even at temperature 0, an answer depends on how much of the prompt the
# server already has cached: a cached prefix changes how the rest is batched,
# so the arithmetic differs slightly and a near-tie between two tokens can go
# either way. Each question is therefore answered once in each cache state a
# real user meets:
#   warm   - after a different question: only the system prompt is cached
#   repeat - the same question again: the whole prompt is cached
#   cold   - the first question after the model loads: nothing is cached
CACHE_STATES = ("warm", "repeat", "cold")

REFUSE_BASELINE = "baseline: always refuse"
COPY_BASELINE = "baseline: copy excerpt [1]"
_REFUSAL_TEXT = "I don't know based on the provided excerpts. Please check Dropbox support."


@dataclass
class ModelResult:
    """Counts are over all answers (one per cache state per question), except
    `unstable` and `near_ctx_limit`, which count questions."""

    name: str
    baseline: bool
    complete: bool  # has an answer for every question in every cache state
    questions: int
    answer_score: float  # mean key-fact score over answerable questions (0 for a refusal)
    ci_low: float
    ci_high: float
    citation_rate: float  # answers to answerable questions citing the right article and nothing out of range
    declines: int  # answers to unanswerable questions that correctly decline
    unanswerable: int
    invented: int  # answers containing a bait string
    refused: int  # answers to answerable questions that refuse
    hedged: int
    steps_ok: int
    procedures: int
    unstable: int  # questions whose answers don't all get the same verdict
    cut_off: int  # stopped by max_tokens
    near_ctx_limit: int  # prompt + max_tokens reached the context window
    avg_citations: float  # distinct excerpts cited per answer that doesn't refuse
    noncanonical: int  # answers citing as "(1)" / "Excerpt 1" instead of [1]
    median_seconds: float | None = None  # warm answers only
    iqr_seconds: tuple[float, float] | None = None
    median_seconds_by_state: dict[str, float] | None = None
    provenance: dict[str, Any] | None = None
    passes_gates: bool = False


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _chunk_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def freeze_missing_contexts(questions: list[EvalQuestion], contexts_path: Path, data_dir: Path) -> None:
    """Retrieve and append contexts for questions that don't have one yet.
    Existing entries never change - delete the file to re-freeze everything."""
    known = {entry["question"] for entry in _read_jsonl(contexts_path)}
    missing = [q for q in questions if q.question not in known]
    if not missing:
        return
    with Retriever(data_dir) as retriever, contexts_path.open("a", encoding="utf-8") as f:
        for q in missing:
            chunks = retriever.search(q.question, top_k=CONTEXT_TOP_K)
            refs = [{"id": c.chunk_id, "sha1": _chunk_hash(c.text), "score": round(c.score, 4)} for c in chunks]
            f.write(json.dumps({"question": q.question, "chunks": refs}, ensure_ascii=False) + "\n")
    print(f"Froze contexts for {len(missing)} new question(s) in {contexts_path} - commit it so the test stays fixed.")


def load_frozen_contexts(contexts_path: Path, articles_path: Path) -> dict[str, list[RetrievedChunk]]:
    """Rebuild each question's frozen excerpts from the crawled articles."""
    chunks_by_id = {c.chunk_id: c for article in load_articles(articles_path) for c in chunk_article(article)}
    contexts = {}
    for entry in _read_jsonl(contexts_path):
        resolved = []
        for ref in entry["chunks"]:
            chunk = chunks_by_id.get(ref["id"])
            if chunk is None or _chunk_hash(chunk.text) != ref["sha1"]:
                raise ValueError(
                    f"Frozen chunk {ref['id']} for {entry['question']!r} is missing or changed in {articles_path} "
                    "(a different crawl, or non-default chunk settings?). Delete the contexts file to re-freeze, "
                    "then re-check the key facts and bait in eval/questions.jsonl."
                )
            resolved.append(RetrievedChunk(score=ref["score"], **asdict(chunk)))
        contexts[entry["question"]] = resolved
    return contexts


def validate_dataset(questions: list[EvalQuestion], contexts: dict[str, list[RetrievedChunk]]) -> list[str]:
    """Key facts must be attainable from the excerpts the model sees; bait
    must be something only an invented answer would say."""
    problems = []
    for q in questions:
        excerpts = " ".join(c.text for c in contexts[q.question])
        if q.expected_url is not None:
            if not q.key_facts:
                problems.append(f"{q.question!r}: answerable but has no key_facts")
            normalized = normalize(excerpts)
            problems += [
                f"{q.question!r}: key fact {fact!r} is not in its frozen excerpts"
                for fact in q.key_facts
                if not contains_fact(normalized, fact)
            ]
        for b in q.bait:
            if light_normalize(b) in light_normalize(excerpts):
                problems.append(f"{q.question!r}: bait {b!r} appears in its own excerpts")
            if light_normalize(b) in light_normalize(q.question):
                problems.append(f"{q.question!r}: bait {b!r} appears in the question itself")
    return problems


def _ollama_url() -> str:
    """Ollama's native API, next to the OpenAI-compatible one."""
    return base_url().rstrip("/").removesuffix("/v1")


def fetch_installed_models() -> tuple[str | None, dict[str, dict[str, str]]]:
    """Ollama's version and {tag: digest/quantization/size} for pulled models,
    or (None, {}) if Ollama isn't reachable."""
    native = _ollama_url()
    try:
        version = httpx.get(f"{native}/api/version", timeout=5).json().get("version")
        tags = httpx.get(f"{native}/api/tags", timeout=5).json().get("models", [])
    except httpx.HTTPError:
        return None, {}
    installed = {}
    for m in tags:
        details = m.get("details") or {}
        installed[m["name"]] = {
            "digest": (m.get("digest") or "")[:12],
            "quantization": details.get("quantization_level", ""),
            "parameters": details.get("parameter_size", ""),
        }
    return version, installed


def _full_tag(model: str) -> str:
    return model if ":" in model else f"{model}:latest"


def _answers_path(answers_dir: Path, model: str) -> Path:
    return answers_dir / (model.replace(":", "_").replace("/", "_") + ".jsonl")


def _cache_key(model: str, messages: list[dict[str, str]], cache_state: str) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "cache_state": cache_state}, sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _reload(model: str) -> None:
    """Unload the model and load it again, which empties the server's prompt cache."""
    httpx.post(f"{_ollama_url()}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60).raise_for_status()
    httpx.post(f"{_ollama_url()}/api/generate", json={"model": model}, timeout=600).raise_for_status()


def generate_answers(
    model: str,
    questions: list[EvalQuestion],
    contexts: dict[str, list[RetrievedChunk]],
    answers_dir: Path,
    provenance: dict[str, Any],
) -> None:
    """Answer every question in every cache state, skipping answers already saved.
    Each state is set up just before its (timed) answer, so a resumed run
    still answers in the right state."""
    path = _answers_path(answers_dir, model)
    saved = {record["key"] for record in _read_jsonl(path)}
    todo = []
    for q in questions:
        messages = build_messages(q.question, contexts[q.question])
        for state in CACHE_STATES:
            key = _cache_key(model, messages, state)
            if key not in saved:
                todo.append((q, messages, state, key))
    if not todo:
        print(f"{model}: all {len(questions) * len(CACHE_STATES)} answers already saved")
        return

    # Start from an empty prompt cache, then cache only the system prompt, so
    # the first "warm" answer starts from the same state as every later one.
    print(f"{model}: loading (not timed)...", flush=True)
    _reload(model)
    system_only = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "Reply with OK."}]
    complete(system_only, model, max_tokens=1)

    path.parent.mkdir(parents=True, exist_ok=True)
    for i, (q, messages, state, key) in enumerate(todo, start=1):
        if state == "repeat":
            complete(messages, model, max_tokens=1)  # make sure this exact prompt is the cached one
        elif state == "cold":
            _reload(model)
        c = complete(messages, model)
        record = {
            "key": key,
            "model": model,
            "question": q.question,
            "cache_state": state,
            "answer": c.text,
            "prompt_tokens": c.prompt_tokens,
            "completion_tokens": c.completion_tokens,
            "finish_reason": c.finish_reason,
            "seconds": round(c.seconds, 2),
            **provenance,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{model}] {i}/{len(todo)}  {state:<6} {c.seconds:5.1f}s  {q.question[:60]}", flush=True)


def load_saved_answers(
    model: str,
    questions: list[EvalQuestion],
    contexts: dict[str, list[RetrievedChunk]],
    answers_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    """question -> its saved records, one per cache state, for answers made
    from the *current* prompt and contexts."""
    by_key = {record["key"]: record for record in _read_jsonl(_answers_path(answers_dir, model))}
    records = {}
    for q in questions:
        messages = build_messages(q.question, contexts[q.question])
        found = [by_key[key] for state in CACHE_STATES if (key := _cache_key(model, messages, state)) in by_key]
        if found:
            records[q.question] = found
    return records


def _first_words(text: str, n: int) -> str:
    """The first n words, keeping line breaks (so numbered steps stay steps)."""
    words = list(re.finditer(r"\S+", text))
    return text if len(words) <= n else text[: words[n - 1].end()]


def baseline_answers(
    questions: list[EvalQuestion], contexts: dict[str, list[RetrievedChunk]]
) -> dict[str, dict[str, list[str]]]:
    """Free reference points: a model should beat pasting the top excerpt,
    and declining everything shouldn't be enough to win. They don't depend on
    any cache, so each gets the same answer in every state."""
    n = len(CACHE_STATES)
    copy_top = {q.question: [_first_words(contexts[q.question][0].text, 200) + " [1]"] * n for q in questions}
    return {
        REFUSE_BASELINE: {q.question: [_REFUSAL_TEXT] * n for q in questions},
        COPY_BASELINE: copy_top,
    }


def bootstrap_ci(values: list[float]) -> tuple[float, float]:
    """95% interval for the mean, resampling questions with replacement."""
    if not values:
        return 0.0, 0.0
    rng = np.random.default_rng(0)
    samples = rng.choice(np.asarray(values), size=(BOOTSTRAP_RESAMPLES, len(values)), replace=True)
    low, high = np.percentile(samples.mean(axis=1), [2.5, 97.5])
    return float(low), float(high)


def _verdict(s: AnswerScore) -> tuple[Any, ...]:
    """Everything the report counts about one answer."""
    return (s.answer_score, s.citation_ok, s.refusal, s.declined_correctly, s.steps_ok, bool(s.bait_hits))


def summarize(
    name: str,
    questions: list[EvalQuestion],
    contexts: dict[str, list[RetrievedChunk]],
    answers: dict[str, list[str]],
    records: dict[str, list[dict[str, Any]]] | None = None,
) -> ModelResult:
    """Score one model's (or baseline's) answers - several per question - and aggregate them."""
    scores: dict[str, list[AnswerScore]] = {}
    for q in questions:
        if q.question in answers:
            urls = [c.url for c in contexts[q.question]]
            scores[q.question] = [
                score_answer(answer, urls, q.expected_url, q.key_facts, q.procedure, q.bait)
                for answer in answers[q.question]
            ]

    def pooled(questions_kept: list[EvalQuestion]) -> list[AnswerScore]:
        return [s for q in questions_kept for s in scores.get(q.question, [])]

    answerable_qs = [q for q in questions if q.expected_url is not None and q.question in scores]
    answerable = pooled(answerable_qs)
    unanswerable = pooled([q for q in questions if q.expected_url is None])
    procedures = pooled([q for q in questions if q.procedure])
    everything = pooled(questions)
    answered = [s for s in everything if s.refusal != "abstained"]
    # One value per question (the mean over its answers), so the CI resamples questions.
    values = [float(np.mean([s.answer_score or 0.0 for s in scores[q.question]])) for q in answerable_qs]
    ci_low, ci_high = bootstrap_ci(values)

    result = ModelResult(
        name=name,
        baseline=records is None,
        complete=all(len(scores.get(q.question, [])) == len(CACHE_STATES) for q in questions),
        questions=len(scores),
        answer_score=float(np.mean(values)) if values else 0.0,
        ci_low=ci_low,
        ci_high=ci_high,
        citation_rate=sum(bool(s.citation_ok) for s in answerable) / len(answerable) if answerable else 0.0,
        declines=sum(bool(s.declined_correctly) for s in unanswerable),
        unanswerable=len(unanswerable),
        invented=sum(bool(s.bait_hits) for s in everything),
        refused=sum(s.refusal == "abstained" for s in answerable),
        hedged=sum(s.refusal == "hedged" for s in answerable),
        steps_ok=sum(bool(s.steps_ok) for s in procedures),
        procedures=len(procedures),
        unstable=sum(len({_verdict(s) for s in question_scores}) > 1 for question_scores in scores.values()),
        cut_off=0,
        near_ctx_limit=0,
        avg_citations=float(np.mean([len(s.citations) for s in answered])) if answered else 0.0,
        noncanonical=sum(s.noncanonical_citation for s in everything),
    )

    if records:
        flat = [r for question_records in records.values() for r in question_records]
        by_state = {state: [r["seconds"] for r in flat if r.get("cache_state") == state] for state in CACHE_STATES}
        warm = by_state["warm"] or [r["seconds"] for r in flat]
        result.median_seconds = statistics.median(warm)
        p25, p75 = np.percentile(warm, [25, 75])
        result.iqr_seconds = (float(p25), float(p75))
        result.median_seconds_by_state = {s: statistics.median(v) for s, v in by_state.items() if v}
        result.cut_off = sum(r["finish_reason"] == "length" for r in flat)
        result.near_ctx_limit = sum(
            max(r["prompt_tokens"] for r in question_records) + MAX_TOKENS >= NUM_CTX
            for question_records in records.values()
        )
        first = flat[0]
        result.provenance = {k: first.get(k, "") for k in ("ollama_version", "digest", "quantization", "parameters")}

    declined_enough = result.unanswerable == 0 or result.declines / result.unanswerable >= DECLINE_GATE
    result.passes_gates = (
        not result.baseline and result.complete and result.citation_rate >= CITATION_GATE and declined_enough
    )
    return result


def pick_winner(results: list[ModelResult]) -> tuple[ModelResult | None, str]:
    """Highest answer score among models passing both gates; if another
    passing model's CI overlaps it and it's clearly faster, that one wins."""
    candidates = sorted((r for r in results if r.passes_gates), key=lambda r: r.answer_score, reverse=True)
    if not candidates:
        return None, (
            f"no model passed both gates (citations >= {CITATION_GATE:.0%} and "
            f">= {DECLINE_GATE:.0%} of unanswerable questions declined)"
        )
    best = candidates[0]
    tied = [r for r in candidates if r.ci_high >= best.ci_low and r.median_seconds is not None]
    fastest = min(tied, key=lambda r: r.median_seconds or 0.0)
    if (
        fastest is not best
        and best.median_seconds
        and fastest.median_seconds is not None
        and fastest.median_seconds <= best.median_seconds * (1 - LATENCY_NOISE)
    ):
        return fastest, (
            f"its answer score is statistically tied with `{best.name}` (overlapping 95% CIs) "
            f"and it is at least {LATENCY_NOISE:.0%} faster"
        )
    return best, "highest answer score among the models that pass both gates"


def _row(r: ModelResult) -> str:
    name = f"*{r.name}*" if r.baseline else f"`{r.name}`"
    if not r.complete:
        name += " (incomplete)"
    speed = "-"
    if r.median_seconds is not None and r.iqr_seconds is not None:
        speed = f"{r.median_seconds:.0f} ({r.iqr_seconds[0]:.0f}-{r.iqr_seconds[1]:.0f})"
    gates = "-" if r.baseline else ("pass" if r.passes_gates else "fail")
    unstable = "-" if r.baseline else f"{r.unstable}/{r.questions}"
    return (
        f"| {name} | {r.answer_score:.2f} ({r.ci_low:.2f}-{r.ci_high:.2f}) | {r.citation_rate:.0%} | "
        f"{r.declines}/{r.unanswerable} | {r.invented} | {r.refused} | {r.hedged} | {r.steps_ok}/{r.procedures} | "
        f"{unstable} | {r.cut_off} | {speed} | {gates} |"
    )


def render_report(
    results: list[ModelResult], winner: ModelResult | None, reason: str, questions: list[EvalQuestion]
) -> str:
    n_answerable = sum(q.expected_url is not None for q in questions)
    n_states = len(CACHE_STATES)
    lines = [
        "# Model benchmark results",
        "",
        f"{len(questions)} questions ({n_answerable} answerable, {len(questions) - n_answerable} unanswerable), "
        f"each answered {n_states} times - once per prompt-cache state ({', '.join(CACHE_STATES)}) - from the "
        f"same frozen top-{CONTEXT_TOP_K} excerpts for every model. Temperature 0, max_tokens {MAX_TOKENS}, "
        f"{NUM_CTX}-token context, CPU only. Generated by `uv run faq-rag bench`.",
        "",
        "| Model | Answer score (95% CI) | Citations ok | Declines | Invented | Refused | Hedged | Steps | Unstable "
        "| Cut off | Median s (IQR) | Gates |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
        *(_row(r) for r in results),
        "",
        f"**Winner: `{winner.name}`** - {reason}." if winner else f"**No winner:** {reason}.",
        "",
        "## What the columns mean",
        "",
        f"Counts are over all answers ({n_states} per question), except **Unstable**, which counts questions.",
        "",
        "- **Answer score**: share of an answerable question's key facts stated in the first 200 words "
        f"(0 if the model refused), averaged over its {n_states} answers and then over questions; the interval "
        "is a bootstrap 95% CI over questions.",
        "- **Citations ok**: answers to answerable questions citing at least one excerpt from the right article "
        f"and no excerpt number that doesn't exist. Gate: >= {CITATION_GATE:.0%}.",
        "- **Declines**: answers to unanswerable questions with an up-front \"I don't know\" and no bait string. "
        f"Gate: >= {DECLINE_GATE:.0%}.",
        "- **Invented**: answers containing a bait string (e.g. a price the corpus never mentions).",
        "- **Refused / Hedged**: answers to answerable questions that refuse outright, or answer after an "
        "up-front hedge.",
        "- **Steps**: answers to how-to questions written as a numbered list.",
        f"- **Unstable**: questions whose {n_states} answers don't all get the same verdict on every column.",
        "- **Cut off**: answers stopped by max_tokens.",
        "- **Median s (IQR)**: wall-clock seconds per warm answer; gaps under "
        f"{LATENCY_NOISE:.0%} are noise on this machine.",
        "",
        "## Diagnostics and provenance",
        "",
    ]
    for r in results:
        if r.baseline:
            continue
        p = r.provenance or {}
        by_state = "/".join(f"{(r.median_seconds_by_state or {}).get(s, 0):.0f}" for s in CACHE_STATES)
        lines.append(
            f"- `{r.name}`: {p.get('parameters', '')} {p.get('quantization', '')}, digest `{p.get('digest', '')}`, "
            f"Ollama {p.get('ollama_version', '')}; median s {'/'.join(CACHE_STATES)}: {by_state}; "
            f"{r.avg_citations:.1f} distinct excerpts cited per answer; {r.noncanonical} non-canonical citation "
            f"formats; {r.near_ctx_limit} prompts near the context limit."
        )
    return "\n".join(lines)


def run_benchmark(
    models: list[str],
    questions_path: Path,
    contexts_path: Path,
    data_dir: Path,
    results_path: Path,
    limit: int | None = None,
) -> None:
    questions = load_questions(questions_path)
    freeze_missing_contexts(questions, contexts_path, data_dir)
    contexts = load_frozen_contexts(contexts_path, data_dir / "articles.jsonl")
    problems = validate_dataset(questions, contexts)
    if problems:
        raise ValueError("Fix these in the eval dataset first:\n  " + "\n  ".join(problems))
    if limit is not None:
        questions = questions[:limit]

    answers_dir = data_dir / "bench" / "answers"
    version, installed = fetch_installed_models()
    if version is None:
        print("Ollama isn't reachable - scoring previously saved answers only.")
    for model in models:
        tag = _full_tag(model)
        if tag in installed:
            generate_answers(model, questions, contexts, answers_dir, {"ollama_version": version, **installed[tag]})
        elif version is not None:
            print(f"{model}: not pulled (`ollama pull {model}`) - using saved answers only.")

    results = []
    for model in models:
        records = load_saved_answers(model, questions, contexts, answers_dir)
        if not records:
            print(f"{model}: no saved answers - left out of the report.")
            continue
        answers = {question: [r["answer"] for r in question_records] for question, question_records in records.items()}
        results.append(summarize(model, questions, contexts, answers, records))
    results.sort(key=lambda r: r.answer_score, reverse=True)
    results += [summarize(name, questions, contexts, answers) for name, answers in baseline_answers(questions, contexts).items()]

    winner, reason = pick_winner(results)
    report = render_report(results, winner, reason, questions)
    print("\n" + report)
    if limit is None:
        results_path.write_text(report + "\n", encoding="utf-8")
        print(f"\nWrote {results_path}")
