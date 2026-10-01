import pytest

from faq_rag.scoring import (
    bait_hits,
    classify_refusal,
    extract_citations,
    has_noncanonical_citation,
    has_numbered_steps,
    key_fact_hits,
    score_answer,
)

URLS = [
    "https://help.dropbox.com/a",
    "https://help.dropbox.com/expected",
    "https://help.dropbox.com/b",
]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("See [1] and [3].", [1, 3]),
        ("Both say so [1][2].", [1, 2]),
        ("As noted [1, 2].", [1, 2]),
        ("Covered in [1-3].", [1, 2, 3]),
        ("Covered in [2–4].", [2, 3, 4]),
        ("Per [Excerpt 2].", [2]),
        ("Per [excerpts 1, 3].", [1, 3]),
        ("Link style [1](https://help.dropbox.com/x).", [1]),
        ("Also [1 and 4].", [1, 4]),
        ("No citations here. 1. Log in", []),
        # bracket numbers that open a line are list markers / reference lists, not inline citations
        ("[1] Log in to dropbox.com\n2. Click Save", []),
        ("Steps done.\n\nCited Excerpts:\n[1] How to delete (https://x)\n[2] Other", []),
        ("  - [2] a bullet that starts with a bracket", []),
        ("[1][2] Both say so.", []),
        ("Click **Save** [1].\n[2] Some list marker", [1]),
        ("1. Log in to dropbox.com [3]", [3]),
    ],
)
def test_extract_citations(text, expected):
    assert extract_citations(text) == expected


def test_noncanonical_citations_are_flagged_not_parsed():
    assert extract_citations("As excerpt (2) explains") == []
    assert has_noncanonical_citation("As excerpt (2) explains")
    assert has_noncanonical_citation("Excerpt 3 says to log in")
    assert not has_noncanonical_citation("Per [Excerpt 3], log in")
    assert not has_noncanonical_citation("1) Log in\n2) Click Settings")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I don't know based on the provided excerpts. Please contact Dropbox support.", "abstained"),
        ("I don’t know.", "abstained"),
        ("The excerpts don't mention pricing for Dropbox Plus.", "abstained"),
        ("The provided excerpts do not contain information about a VPN.", "abstained"),
        (
            "I don't know the exact date, but here's how to delete your account:\n"
            "1. Log in to dropbox.com\n2. Click your avatar\n3. Click Settings [1]",
            "hedged",
        ),
        (
            # a faithful answer that repeats the article's own "can't find" advice
            "To change your email, open Settings [1]. Dropbox doesn't have access to your password [2]. "
            "If you can't find the verification email, contact Dropbox Support.",
            "answered",
        ),
        (
            # refusal phrase only appears after a real answer
            "Log in and open Settings [1]. Then click Delete account [1]. I don't know if this applies to teams.",
            "answered",
        ),
        # passive meta-statements about the excerpts (real granite4:3b phrasings)
        (
            "The Dropbox Plus subscription cost is not explicitly stated in the provided excerpts. "
            "However, it mentions a 25% discount on ASUS computers [2].",
            "abstained",
        ),
        (
            "The phone number for Dropbox customer support is not provided in the given excerpts. "
            "However, you can use the customer service portal [1].",
            "abstained",
        ),
        ("The user's question is not directly answerable from the provided excerpts.", "abstained"),
        ("There's no mention of a student discount in the excerpts.", "abstained"),
        ("A student discount isn't mentioned in the excerpts.", "abstained"),
        ("I'm sorry, but the weather is outside the scope of the Dropbox Help Center.", "abstained"),
        ("I can only help with questions about Dropbox.", "abstained"),
        # real phi4-mini:3.8b phrasings
        ("I'm sorry, but I don't have the information you requested. You may want to check a weather service.", "abstained"),
        ("I'm sorry, but I can't provide you with a Python script as requested.", "abstained"),
        # real granite4:3b phrasings from run 3
        (
            "I'm sorry, but I don't have the capability to write a Python script to sort a list of numbers "
            "based on the provided excerpts.",
            "abstained",
        ),
        ("The excerpts provided do not contain information about a student discount for Dropbox.", "abstained"),
        ("The information provided does not pertain to a student discount for Dropbox.", "abstained"),
        # plain negations about the product itself are not refusals
        ("If your payment information doesn't include a billing address, add it under Billing [1].", "answered"),
        ("If your device is not listed in the security page, log in again [1].", "answered"),
        ("Files not found in your Dropbox folder may be in the deleted files page [2].", "answered"),
        ("Your password is not provided by your admin; reset it from the login page [1].", "answered"),
    ],
)
def test_classify_refusal(text, expected):
    assert classify_refusal(text) == expected


def test_long_refusal_without_list_is_hedged():
    text = "I don't know the answer from these excerpts. " + "This part rambles on about related topics. " * 12
    assert classify_refusal(text) == "hedged"


def test_numbered_steps():
    assert has_numbered_steps("1. Log in\n2. Click your avatar")
    assert has_numbered_steps("Step 1: Log in\nStep 2: Click Settings")
    assert has_numbered_steps("**1.** Log in\n**2.** Click Settings")
    assert not has_numbered_steps("Log in and click Delete account.")
    assert not has_numbered_steps("1. Log in to dropbox.com")


def test_key_facts_normalize_markdown_case_and_punctuation():
    assert key_fact_hits("Click **Manage account**.", ["Manage account"]) == [True]
    assert key_fact_hits("then manage account, then...", ["Manage Account"]) == [True]
    # parser artifact in the source text: no space after bold
    assert key_fact_hits("Click **Admin console**in the sidebar", ["admin console"]) == [True]


def test_key_facts_alternatives_and_whole_words():
    assert key_fact_hits("You can delete your account.", ["Delete account|delete your account"]) == [True]
    assert key_fact_hits("Link up to 3 devices [1].", ["3 devices|three devices"]) == [True]
    assert key_fact_hits("Use three devices.", ["3 devices|three devices"]) == [True]
    # a citation number must not count as the fact "3 devices"
    assert key_fact_hits("See [3]. Devices vary.", ["3 devices"]) == [False]
    # whole-word: "plus" must not match inside "surplus"
    assert key_fact_hits("There is a surplus.", ["plus"]) == [False]


def test_key_facts_only_count_within_the_window():
    padding = "word " * 250
    assert key_fact_hits(padding + "Manage account", ["Manage account"]) == [False]


def test_bait_hits_keep_punctuation():
    assert bait_hits("Dropbox Plus costs $11.99 per month.", ["$", "per month", "/year"]) == ["$", "per month"]
    assert bait_hits("The excerpts don't mention the price.", ["$", "per month"]) == []


def test_score_answerable_with_correct_citation():
    score = score_answer(
        "1. Click **Manage account** [2]\n2. Click Cancel [2]",
        URLS,
        expected_url="https://help.dropbox.com/expected",
        key_facts=["Manage account", "Delete account"],
        procedure=True,
        bait=[],
    )
    assert score.refusal == "answered"
    assert score.citations == [2]
    assert score.citation_ok is True
    assert score.key_fact_hits == [True, False]
    assert score.answer_score == 0.5
    assert score.steps_ok is True


def test_score_out_of_range_citation_fails_citation_check():
    score = score_answer(
        "Click Manage account [2][7].",
        URLS,
        expected_url="https://help.dropbox.com/expected",
        key_facts=["Manage account"],
        procedure=False,
        bait=[],
    )
    assert score.invalid_citations == [7]
    assert score.citation_ok is False
    assert score.steps_ok is None


def test_score_refusing_an_answerable_question_scores_zero():
    score = score_answer(
        "I don't know. Please contact Dropbox support.",
        URLS,
        expected_url="https://help.dropbox.com/expected",
        key_facts=["Manage account"],
        procedure=False,
        bait=[],
    )
    assert score.refusal == "abstained"
    assert score.answer_score == 0.0
    assert score.citation_ok is False


@pytest.mark.parametrize(
    ("answer", "declined"),
    [
        ("The excerpts don't mention the price of Dropbox Plus.", True),
        ("I don't know for sure, but Dropbox Plus is about $12 per month.", False),  # hedge that invents
        ("Dropbox Plus has several features [1].", False),  # answered instead of declining
    ],
)
def test_score_unanswerable(answer, declined):
    score = score_answer(answer, URLS, expected_url=None, key_facts=[], procedure=False, bait=["$", "per month"])
    assert score.declined_correctly is declined
    assert score.answer_score is None
