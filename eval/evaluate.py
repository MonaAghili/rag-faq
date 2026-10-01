#!/usr/bin/env python3
"""Standalone entry point for `uv run python eval/evaluate.py`.

Equivalent to `uv run faq-rag eval` - kept as a plain script too since eval
harnesses are often run/edited independently of the CLI.
"""

from pathlib import Path

from faq_rag.evaluator import run_evaluation

if __name__ == "__main__":
    run_evaluation(data_dir=Path("data"), questions_path=Path(__file__).parent / "questions.jsonl")
