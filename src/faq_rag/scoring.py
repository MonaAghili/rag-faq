"""Deterministic answer checks for the model benchmark (no LLM judge).

Every function here is pure - text in, verdict out - so each check can be
unit-tested with tricky strings, and re-scoring saved answers after fixing a
check never requires regenerating them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

KEY_FACT_WINDOW_WORDS = 200  # only the start of an answer counts, so padding can't game recall
ABSTAIN_MAX_WORDS = 60

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', " ": " "})

# [1]  [1][2]  [1, 2]  [1-3]  [1 and 2]  [Excerpt 1]  [1](url) - the "(url)" part is ignored.
_CITATION_RE = re.compile(
    r"\[\s*(?:excerpts?\s*)?(\d+(?:\s*(?:,|-|–|and)\s*(?:excerpts?\s*)?\d+)*)\s*\]",
    re.IGNORECASE,
)
# "(1)" and a bare "Excerpt 1" are logged but not parsed: they collide with list markers.
_NONCANONICAL_RE = re.compile(r"\(\s*\d+\s*\)|(?<!\[)\bexcerpt\s+\d+\b", re.IGNORECASE)

# Statements *about the excerpts* ("the excerpts don't mention X", "X is not
# explicitly stated in the provided excerpts") or first-person declines. Help
# articles themselves say things like "if you can't find it", so plain
# negations about the product never count - only these meta forms do.
_EXCERPTS = r"(?:the |these |any of the )?(?:provided |given |available |above )?(?:excerpts?|context|documents?|sources?|information)"
_REFUSAL_RE = re.compile(
    r"\bi (?:do not|don't|dont) (?:know|have (?:enough |any |the |that |this )?"
    r"(?:information|details|info|data|capability|ability|knowledge))"
    r"|\bi(?:'m| am) (?:not sure|unable to|not able to)"
    r"|\bi (?:cannot|can't|could not|couldn't) (?:find|answer|determine|tell|say|confirm|see|help|provide|give"
    r"|assist|write|offer|share)"
    r"|\bi (?:do not|don't) see\b"
    r"|\bi can only (?:help|answer|assist)"
    r"|\b(?:outside|beyond) (?:the |my )?scope\b"
    # "information" only with a participle: "your billing information doesn't include" is product text
    r"|\b(?:(?:excerpts?|documents?|context|sources?)(?: provided| given| above)?|information (?:provided|given|above))"
    r" (?:do not|don't|does not|doesn't) (?:contain|mention|cover|include|provide|say|specify|address|explain|describe"
    r"|discuss|pertain|relate)"
    r"|\bthere(?: is|'s| are) no (?:information|details|mention)"
    rf"|(?:\bnot|n't|\bno) (?:\w+ ){{0,4}}(?:covered|mentioned|addressed|included|specified|stated|provided|given|listed"
    rf"|found|available|described|explained|discussed|answerable|answered|contained) (?:in|within|by|from) {_EXCERPTS}"
)
_LIST_LINE_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+\S", re.MULTILINE)
_STEP_LINE_RE = re.compile(r"^\s*(?:\*\*)?(?:step\s+)?\d+[.):]", re.MULTILINE | re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class AnswerScore:
    refusal: str  # "answered" | "hedged" | "abstained"
    citations: list[int]
    invalid_citations: list[int]
    noncanonical_citation: bool
    bait_hits: list[str]
    steps_ok: bool | None  # None unless the question is a how-to
    key_fact_hits: list[bool] = field(default_factory=list)  # empty for unanswerable questions
    recall: float | None = None
    answer_score: float | None = None  # recall, or 0 if abstained; None for unanswerable
    citation_ok: bool | None = None  # cites the expected article and nothing out of range
    declined_correctly: bool | None = None  # only for unanswerable questions


def light_normalize(text: str) -> str:
    """Lowercase, straight quotes, plain spaces - keeps punctuation like '$'."""
    return text.translate(_QUOTES).lower()


def normalize(text: str) -> str:
    """For whole-word fact matching: also drops markdown/punctuation, so
    '**Manage account**' and 'manage account.' compare equal."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", light_normalize(text)).split())


def _opens_line(text: str, pos: int) -> bool:
    """True if nothing but whitespace, bullets, bold markers or other
    brackets precede `pos` on its line."""
    prefix = text[text.rfind("\n", 0, pos) + 1 : pos]
    return not _CITATION_RE.sub("", prefix).strip(" \t*-•")


def extract_citations(text: str) -> list[int]:
    """Inline citations only. A bracket number that opens a line ("[1] Log in",
    or a trailing "Sources:\\n[1] Title" list) is a list marker or reference
    list, not a citation of a statement - small models do both."""
    cited: set[int] = set()
    for match in _CITATION_RE.finditer(text):
        if _opens_line(text, match.start()):
            continue
        for part in re.split(r"\s*(?:,|and)\s*", match.group(1), flags=re.IGNORECASE):
            numbers = [int(n) for n in re.findall(r"\d+", part)]
            if len(numbers) == 2 and numbers[0] <= numbers[1] <= numbers[0] + 10:  # a range like 1-3
                cited.update(range(numbers[0], numbers[1] + 1))
            else:
                cited.update(numbers)
    return sorted(cited)


def strip_citations(text: str) -> str:
    return _CITATION_RE.sub(" ", text)


def has_noncanonical_citation(text: str) -> bool:
    return bool(_NONCANONICAL_RE.search(strip_citations(text)))


def classify_refusal(text: str) -> str:
    """'abstained' only if a refusal phrase opens the answer AND the answer is
    short with no list. A refusal phrase later on, or followed by a real
    answer, isn't a refusal - help articles themselves say things like
    "if you can't find it, contact Dropbox Support"."""
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(light_normalize(text).strip()) if s.strip()]
    if not _REFUSAL_RE.search(" ".join(sentences[:2])):
        return "answered"
    if len(text.split()) < ABSTAIN_MAX_WORDS and not _LIST_LINE_RE.search(text):
        return "abstained"
    return "hedged"


def has_numbered_steps(text: str) -> bool:
    return len(_STEP_LINE_RE.findall(text)) >= 2


def contains_fact(normalized_text: str, fact: str) -> bool:
    """Whole-word match of any '|'-separated alternative in already-normalized text."""
    haystack = f" {normalized_text} "
    return any(f" {normalize(alt)} " in haystack for alt in fact.split("|") if normalize(alt))


def key_fact_hits(text: str, key_facts: list[str]) -> list[bool]:
    """Which key facts the answer states, within its first KEY_FACT_WINDOW_WORDS words."""
    window = " ".join(strip_citations(text).split()[:KEY_FACT_WINDOW_WORDS])
    normalized = normalize(window)
    return [contains_fact(normalized, fact) for fact in key_facts]


def bait_hits(text: str, bait: list[str]) -> list[str]:
    """Strings only an invented answer would contain (e.g. '$' for a price
    the corpus never mentions). Plain substring match, punctuation kept."""
    haystack = light_normalize(text)
    return [b for b in bait if light_normalize(b) in haystack]


def score_answer(
    answer: str,
    context_urls: list[str],
    expected_url: str | None,
    key_facts: list[str],
    procedure: bool,
    bait: list[str],
) -> AnswerScore:
    """Score one answer. context_urls[i] is the URL of excerpt [i+1]."""
    citations = extract_citations(answer)
    invalid = [n for n in citations if not 1 <= n <= len(context_urls)]
    score = AnswerScore(
        refusal=classify_refusal(answer),
        citations=citations,
        invalid_citations=invalid,
        noncanonical_citation=has_noncanonical_citation(answer),
        bait_hits=bait_hits(answer, bait),
        steps_ok=has_numbered_steps(answer) if procedure else None,
    )

    if expected_url is None:
        score.declined_correctly = score.refusal != "answered" and not score.bait_hits
        return score

    score.key_fact_hits = key_fact_hits(answer, key_facts)
    score.recall = sum(score.key_fact_hits) / len(score.key_fact_hits) if score.key_fact_hits else 0.0
    score.answer_score = 0.0 if score.refusal == "abstained" else score.recall
    cites_expected = any(context_urls[n - 1] == expected_url for n in citations if 1 <= n <= len(context_urls))
    score.citation_ok = cites_expected and not invalid
    return score
