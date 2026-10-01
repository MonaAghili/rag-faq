"""Crawl help.dropbox.com: sitemap -> filtered article URLs -> data/articles.jsonl.

Resumable by design: articles.jsonl is appended to as each page is parsed, and
URLs already saved by the current parser version are skipped on the next run
(pages saved by an older parser version are re-fetched).
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from faq_rag.parser import PARSER_VERSION, parse_article

SITEMAP_URL = "https://help.dropbox.com/sitemap.xml"

# help.dropbox.com returns HTTP 202 with an empty body to requests that don't
# look like a browser, so we spoof a normal set of browser headers.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

LOCALE_SEGMENT_RE = re.compile(r"^[a-z]{2}-[a-z]{2}$")

MAX_ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 1.0

# If this many rate-limit-style failures happen in a row, the site is very
# likely in a sustained throttling window (observed in practice: it can take
# much longer to recover than any single URL's retry budget). Better to stop
# and let the next `crawl` invocation pick up the rest than to grind through
# every remaining URL at ~15s of wasted retries each.
MAX_CONSECUTIVE_RATE_LIMIT_FAILURES = 10


NOT_FOUND_REASON = "404 not found"
CHALLENGE_REASON = "blocked by bot protection (AWS WAF challenge)"


class FetchOutcome:
    """Result of fetching one URL: exactly one of the two fields is set."""

    def __init__(self, html: str | None = None, fail_reason: str | None = None):
        self.html = html
        self.fail_reason = fail_reason


def load_sitemap_urls(cache_path: Path, client: httpx.Client) -> list[str]:
    """Return every <loc> in help.dropbox.com's sitemap, caching the raw file
    to disk (it's ~4.6MB) so re-runs don't re-download it."""
    if cache_path.exists():
        raw = cache_path.read_bytes()
    else:
        response = client.get(SITEMAP_URL, headers=BROWSER_HEADERS, follow_redirects=True)
        response.raise_for_status()
        raw = response.content
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(raw)

    soup = BeautifulSoup(raw, "xml")
    return [loc.get_text(strip=True) for loc in soup.find_all("loc")]


def filter_article_urls(urls: list[str]) -> list[str]:
    """Keep only English help-center articles: drop locale-prefixed URLs
    (e.g. /ja-jp/...) and category landing pages (fewer than 2 path segments).
    The sitemap groups URLs alphabetically by category, so the result is
    interleaved across categories - that way even a partial or rate-limited
    crawl samples every category instead of exhausting them alphabetically."""
    kept = []
    for url in urls:
        segments = urlparse(url).path.strip("/").split("/")
        if not segments or not segments[0]:
            continue
        if LOCALE_SEGMENT_RE.match(segments[0]):
            continue
        if len(segments) < 2:
            continue
        kept.append(url)
    return _interleave_by_category(kept)


def _interleave_by_category(urls: list[str]) -> list[str]:
    by_category: dict[str, list[str]] = {}
    for url in urls:
        category = urlparse(url).path.strip("/").split("/")[0]
        by_category.setdefault(category, []).append(url)

    queues = list(by_category.values())
    interleaved = []
    while queues:
        next_round = []
        for queue in queues:
            interleaved.append(queue.pop(0))
            if queue:
                next_round.append(queue)
        queues = next_round
    return interleaved


def load_saved_versions(articles_path: Path) -> dict[str, int]:
    """URL -> newest parser version it was saved with."""
    if not articles_path.exists():
        return {}
    versions: dict[str, int] = {}
    with articles_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                version = record.get("parser_version", 1)
                versions[record["url"]] = max(version, versions.get(record["url"], 0))
    return versions


def plan_fetch(urls: list[str], saved_versions: dict[str, int]) -> list[str]:
    """What to fetch, in order: pages saved by an older parser first (so a
    parser fix reaches the existing corpus before it grows), then pages never
    crawled. Pages saved by the current parser are skipped."""
    outdated = [u for u in urls if u in saved_versions and saved_versions[u] < PARSER_VERSION]
    missing = [u for u in urls if u not in saved_versions]
    return outdated + missing


async def fetch_html(client: httpx.AsyncClient, url: str) -> FetchOutcome:
    """Fetch one page, retrying with exponential backoff on the bot-protection
    response (202, or an empty body under any status). 404 is a permanent
    failure - a handful of sitemap entries point at pages that no longer exist."""
    last_status: int | None = None
    for attempt in range(MAX_ATTEMPTS):
        if attempt > 0:
            await asyncio.sleep(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))

        try:
            response = await client.get(url, headers=BROWSER_HEADERS, follow_redirects=True)
        except httpx.HTTPError as exc:
            last_status = None
            if attempt == MAX_ATTEMPTS - 1:
                return FetchOutcome(fail_reason=f"request error: {exc}")
            continue

        if response.status_code == 404:
            return FetchOutcome(fail_reason=NOT_FOUND_REASON)
        if response.headers.get("x-amzn-waf-action") == "challenge":
            # A JavaScript challenge a plain HTTP client can't pass (and this
            # crawler won't try to): retrying only adds requests that keep the
            # block going, so give up on the spot.
            return FetchOutcome(fail_reason=CHALLENGE_REASON)
        if response.status_code == 200 and response.text.strip():
            return FetchOutcome(html=response.text)
        last_status = response.status_code

    return FetchOutcome(fail_reason=f"gave up after {MAX_ATTEMPTS} attempts (last status {last_status})")


async def run_crawl(
    urls: list[str],
    articles_path: Path,
    concurrency: int,
    delay: float,
) -> dict[str, int | bool]:
    """Fetch, parse, and append each URL not already saved by the current
    parser (re-fetched pages are appended; readers keep the newest record).
    Prints one progress line per URL and returns final saved/skipped/failed counts."""
    to_fetch = plan_fetch(urls, load_saved_versions(articles_path))
    skipped = len(urls) - len(to_fetch)

    saved_count = 0
    failed_count = 0
    consecutive_rate_limit_failures = 0
    aborted = False
    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(concurrency)
    total = len(to_fetch)

    async with httpx.AsyncClient(timeout=30.0) as client:

        async def process(i: int, url: str) -> None:
            nonlocal saved_count, failed_count, consecutive_rate_limit_failures, aborted
            async with semaphore:
                # Checked *after* acquiring the semaphore (not before): asyncio.gather
                # starts every coroutine immediately, so a check before the semaphore
                # would pass for nearly all of them before any failures even happen.
                # Gating here means at most `concurrency` extra requests run once
                # the breaker trips, instead of every remaining URL.
                if aborted:
                    return
                await asyncio.sleep(delay)
                outcome = await fetch_html(client, url)

            if outcome.html is None:
                failed_count += 1
                print(f"[{i}/{total}] FAILED  {url}  ({outcome.fail_reason})")
                if outcome.fail_reason == CHALLENGE_REASON:
                    if not aborted:
                        print(
                            "\nDropbox's bot protection is challenging requests - stopping now. Articles "
                            "saved so far are safe; run `faq-rag crawl` again later to continue."
                        )
                    aborted = True
                elif outcome.fail_reason != NOT_FOUND_REASON:
                    consecutive_rate_limit_failures += 1
                    if consecutive_rate_limit_failures >= MAX_CONSECUTIVE_RATE_LIMIT_FAILURES:
                        if not aborted:
                            print(
                                f"\n{consecutive_rate_limit_failures} rate-limit-style failures in a row - "
                                "the site is likely throttling this run. Stopping early; articles saved so "
                                "far are safe, run `faq-rag crawl` again later to pick up where this left off."
                            )
                        aborted = True
                return

            consecutive_rate_limit_failures = 0
            article = parse_article(outcome.html, url)
            if article is None:
                failed_count += 1
                print(f"[{i}/{total}] FAILED  {url}  (could not find expected article structure)")
                return

            async with write_lock:
                with articles_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(article), ensure_ascii=False) + "\n")
            saved_count += 1
            print(f"[{i}/{total}] saved   {url}")

        await asyncio.gather(*(process(i, url) for i, url in enumerate(to_fetch, start=1)))

    return {"saved": saved_count, "skipped": skipped, "failed": failed_count, "aborted_early": aborted}
