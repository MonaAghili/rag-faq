"""Retrieval-quality evaluation: hit@1/3/5 and MRR at the article level.

A chunk counts as a hit if its `url` matches the question's expected_url,
regardless of which section/chunk of that article it came from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from faq_rag.retriever import Retriever

EVAL_TOP_K = 5


@dataclass
class EvalQuestion:
    question: str
    expected_url: str | None  # None marks a question the help center can't answer
    # Used only by the model benchmark (see benchmark.py / scoring.py):
    key_facts: list[str] = field(default_factory=list)  # "a|b" = either alternative counts
    procedure: bool = False  # a how-to question, so the answer should be numbered steps
    bait: list[str] = field(default_factory=list)  # strings only an invented answer would contain


@dataclass
class QuestionResult:
    question: str
    expected_url: str | None
    retrieved_urls: list[str]
    rank: int | None  # 1-based rank of the first chunk whose url matches, or None


def load_questions(questions_path: Path) -> list[EvalQuestion]:
    questions = []
    with questions_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                questions.append(
                    EvalQuestion(
                        question=data["question"],
                        expected_url=data.get("expected_url"),
                        key_facts=data.get("key_facts", []),
                        procedure=data.get("procedure", False),
                        bait=data.get("bait", []),
                    )
                )
    return questions


def evaluate_question(retriever: Retriever, question: EvalQuestion) -> QuestionResult:
    chunks = retriever.search(question.question, top_k=EVAL_TOP_K)
    retrieved_urls = [c.url for c in chunks]

    rank = None
    if question.expected_url is not None:
        for i, url in enumerate(retrieved_urls, start=1):
            if url == question.expected_url:
                rank = i
                break

    return QuestionResult(
        question=question.question,
        expected_url=question.expected_url,
        retrieved_urls=retrieved_urls,
        rank=rank,
    )


def compute_metrics(results: list[QuestionResult]) -> dict[str, float]:
    """Computed only over questions that have an expected_url (a question
    with expected_url=None is excluded from ranking, e.g. for a
    deliberately-unanswerable question)."""
    answerable = [r for r in results if r.expected_url is not None]
    if not answerable:
        return {"hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0, "mrr": 0.0}

    def hit_at(k: int) -> float:
        return sum(1 for r in answerable if r.rank is not None and r.rank <= k) / len(answerable)

    mrr = sum((1 / r.rank) if r.rank is not None else 0.0 for r in answerable) / len(answerable)
    return {"hit@1": hit_at(1), "hit@3": hit_at(3), "hit@5": hit_at(5), "mrr": mrr}


def run_evaluation(data_dir: Path, questions_path: Path) -> None:
    """Print hit@k/MRR over the answerable questions and list any misses."""
    questions = load_questions(questions_path)
    with Retriever(data_dir) as retriever:
        results = [evaluate_question(retriever, q) for q in questions]

    answerable_count = sum(1 for r in results if r.expected_url is not None)
    metrics = compute_metrics(results)
    print(f"Questions: {len(questions)} ({answerable_count} answerable, {len(questions) - answerable_count} expected-unanswerable)")
    print(
        f"hit@1={metrics['hit@1']:.2f}  hit@3={metrics['hit@3']:.2f}  "
        f"hit@5={metrics['hit@5']:.2f}  MRR={metrics['mrr']:.3f}"
    )

    misses = [r for r in results if r.expected_url is not None and r.rank is None]
    if misses:
        print(f"\nMissed ({len(misses)}):")
        for r in misses:
            print(f"  - {r.question!r}\n      expected {r.expected_url}\n      got {r.retrieved_urls[:3]}")
