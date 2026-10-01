import asyncio
import json

import httpx

from faq_rag.crawler import CHALLENGE_REASON, NOT_FOUND_REASON, fetch_html, load_saved_versions, plan_fetch
from faq_rag.parser import PARSER_VERSION

A, B, C, D = (f"https://help.dropbox.com/cat/{name}" for name in "abcd")


def test_plan_fetch_refetches_outdated_pages_first_then_new_ones():
    saved = {A: PARSER_VERSION, B: PARSER_VERSION - 1}
    # A is current (skipped), B was parsed by an older parser, C and D are new
    assert plan_fetch([A, C, B, D], saved) == [B, C, D]


def test_load_saved_versions_treats_unversioned_records_as_version_1(tmp_path):
    path = tmp_path / "articles.jsonl"
    records = [
        {"url": A},  # saved before versioning existed
        {"url": B, "parser_version": 1},
        {"url": B, "parser_version": 2},  # B was re-crawled later
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    assert load_saved_versions(path) == {A: 1, B: 2}


def _fetch_with(response: httpx.Response) -> tuple[object, int]:
    """Run fetch_html against a fake server that always sends `response`."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return response

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await fetch_html(client, A)

    return asyncio.run(run()), calls


def test_bot_challenge_gives_up_after_one_request():
    outcome, calls = _fetch_with(httpx.Response(202, headers={"x-amzn-waf-action": "challenge"}, text="<html>js</html>"))
    assert outcome.fail_reason == CHALLENGE_REASON
    assert calls == 1


def test_not_found_is_not_retried():
    outcome, calls = _fetch_with(httpx.Response(404))
    assert outcome.fail_reason == NOT_FOUND_REASON
    assert calls == 1


def test_a_normal_page_is_returned():
    outcome, _ = _fetch_with(httpx.Response(200, text="<html>article</html>"))
    assert outcome.html == "<html>article</html>"
