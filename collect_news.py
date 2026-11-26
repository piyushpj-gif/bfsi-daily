from datetime import datetime, timedelta, timezone
import html
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

import feedparser


FEEDS_FILE = Path(__file__).resolve().parent / "feeds.json"
USER_AGENT = "Mozilla/5.0 (compatible; bfsi-daily/1.0)"
FEED_TIMEOUT_SECONDS = 15
MAX_ITEMS_PER_SOURCE = 25
MAX_SNIPPET_LENGTH = 300
JUNK_TITLE_PATTERNS = (
    re.compile(r"news by cnbc tv18 on tradingview", re.IGNORECASE),
    re.compile(r"^news by .+ on tradingview\b", re.IGNORECASE),
)


def load_feeds(feeds_path: Path) -> list[dict]:
    """Load RSS feed definitions from feeds.json."""
    with feeds_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def fetch_feed(url: str, timeout: int = FEED_TIMEOUT_SECONDS) -> feedparser.FeedParserDict:
    """Download an RSS feed with a timeout and parse it using feedparser."""
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        raw_content = response.read()
    return feedparser.parse(raw_content)


def get_entry_datetime(entry: dict) -> datetime | None:
    """Return the entry's UTC publication datetime, or None if missing."""
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed is None:
        return None
    return datetime(*parsed[:6], tzinfo=timezone.utc)


def clean_title(raw_title: str) -> str:
    """Unescape HTML entities and collapse extra whitespace in a title."""
    return " ".join(html.unescape(raw_title or "").split())


def clean_snippet(entry: dict, max_length: int = MAX_SNIPPET_LENGTH) -> str:
    """Extract a plain-text description with HTML tags removed and capped length."""
    raw_summary = entry.get("summary") or entry.get("description") or ""
    if not raw_summary and entry.get("content"):
        raw_summary = entry["content"][0].get("value", "")

    text = html.unescape(raw_summary)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = " ".join(text.split())

    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def split_aggregator_title(raw_title: str) -> tuple[str, str]:
    """Split a Google News-style title into (clean_title, publisher).

    Google News titles follow the pattern "Story headline - Publisher Name".
    Returns the headline and publisher as separate strings.
    If no " - " separator is found, returns the full title and an empty publisher.
    """
    # rsplit so only the last " - " is treated as the separator (handles dashes in headlines)
    parts = raw_title.rsplit(" - ", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return raw_title, ""


def is_redundant_snippet(snippet: str, title: str, publisher: str) -> bool:
    """Return True if the snippet, after cleaning, only repeats the title or the publisher.

    Comparison is case-insensitive and ignores leading/trailing whitespace.
    """
    if not snippet:
        return True
    normalised = snippet.strip().lower()
    if normalised == title.strip().lower():
        return True
    if publisher and normalised == publisher.strip().lower():
        return True
    return False


def is_junk_title(title: str) -> bool:
    """Return True for empty titles or obvious boilerplate/aggregator junk."""
    if not title:
        return True
    return any(pattern.search(title) for pattern in JUNK_TITLE_PATTERNS)


def collect_news(
    feeds: list[dict] | None = None,
    hours: int = 24,
) -> tuple[dict[str, list[str]], list[dict]]:
    """Fetch feeds, filter recent unique items, and return grouped titles plus structured items."""
    if feeds is None:
        feeds = load_feeds(FEEDS_FILE)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    seen_links: set[str] = set()
    seen_titles: set[str] = set()
    results_by_source: dict[str, list[str]] = {}
    items: list[dict] = []
    next_id = 1

    for feed_info in feeds:
        source_name = feed_info["name"]
        source_type = feed_info.get("type", "direct")
        label = f"{source_name} [{source_type}]" if source_type == "aggregator" else source_name
        url = feed_info["url"]
        results_by_source[label] = []

        try:
            parsed_feed = fetch_feed(url)
        except Exception as exc:
            print(f"[Warning] Skipping {source_name} ({url}): {exc}")
            continue

        if getattr(parsed_feed, "bozo", False) and not parsed_feed.entries:
            print(f"[Warning] Skipping {source_name} ({url}): {parsed_feed.bozo_exception}")
            continue

        for entry in parsed_feed.entries:
            if len(results_by_source[label]) >= MAX_ITEMS_PER_SOURCE:
                break

            published_at = get_entry_datetime(entry)
            if published_at is not None and published_at < cutoff:
                continue

            title = clean_title(entry.get("title") or "")
            if is_junk_title(title):
                continue

            # For aggregator feeds split "Headline - Publisher" apart.
            if source_type == "aggregator":
                title, publisher = split_aggregator_title(title)
                title = title.strip()
                if is_junk_title(title):
                    continue
            else:
                publisher = ""

            link = (entry.get("link") or "").strip()
            title_key = title.lower()

            if (link and link in seen_links) or title_key in seen_titles:
                continue

            if link:
                seen_links.add(link)
            seen_titles.add(title_key)

            published_value = (
                published_at.isoformat()
                if published_at
                else (entry.get("published") or entry.get("updated") or "").strip()
            )

            snippet = clean_snippet(entry)
            if is_redundant_snippet(snippet, title, publisher):
                snippet = ""

            # Determine item source: for aggregators use the publisher extracted from the title;
            # fall back to the feed name if no publisher suffix was found.
            item_source = publisher if (source_type == "aggregator" and publisher) else source_name

            results_by_source[label].append(title)
            item: dict = {
                "id": next_id,
                "source": item_source,
                "title": title,
                "link": link,
                "published": published_value,
                "snippet": snippet,
            }
            if source_type == "aggregator":
                item["via"] = source_name
            items.append(item)
            next_id += 1

    return results_by_source, items


def main() -> None:
    """Load feeds, collect recent unique articles, and print counts and titles per source."""
    feeds = load_feeds(FEEDS_FILE)
    news_by_source, _ = collect_news(feeds)

    for source_name, titles in news_by_source.items():
        print(f"\n{source_name}: {len(titles)} item(s)")
        for title in titles:
            print(f"  - {title}")


if __name__ == "__main__":
    main()
