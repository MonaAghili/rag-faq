from faq_rag.benchmark import CACHE_STATES, _cache_key, summarize
from faq_rag.evaluator import EvalQuestion
from faq_rag.retriever import RetrievedChunk

URL = "https://help.dropbox.com/plans/add-licenses"
HOW = EvalQuestion("How do I add licenses?", URL, key_facts=["Manage licenses"])
PRICE = EvalQuestion("How much does Plus cost?", None, bait=["$"])
CHUNK = RetrievedChunk(0.9, "plans/add-licenses::000", URL, "Add licenses", "Overview", "Plans", "Click Manage licenses.")
CONTEXTS = {q.question: [CHUNK] for q in (HOW, PRICE)}

GOOD = "Click **Manage licenses** [1]."
REFUSAL = "I don't know."


def test_scores_are_pooled_over_the_answers_in_each_cache_state():
    answers = {
        HOW.question: [GOOD, GOOD, REFUSAL],
        PRICE.question: [REFUSAL, REFUSAL, "It costs $10 a month."],
    }
    result = summarize("model", [HOW, PRICE], CONTEXTS, answers)

    assert result.complete
    assert round(result.answer_score, 3) == 0.667  # 1, 1 and 0 (refused)
    assert round(result.citation_rate, 3) == 0.667
    assert (result.refused, result.declines, result.unanswerable, result.invented) == (1, 2, 3, 1)
    assert result.unstable == 2  # both questions got different verdicts in different states


def test_questions_answered_the_same_way_every_time_are_stable():
    answers = {HOW.question: [GOOD] * 3, PRICE.question: [REFUSAL] * 3}
    assert summarize("model", [HOW, PRICE], CONTEXTS, answers).unstable == 0


def test_each_cache_state_has_its_own_cache_key():
    messages = [{"role": "user", "content": "How do I add licenses?"}]
    assert len({_cache_key("model", messages, state) for state in CACHE_STATES}) == len(CACHE_STATES)
