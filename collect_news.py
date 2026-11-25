from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from urllib.request import Request, urlopen

import feedparser


FEEDS_FILE = Path(__file__).resolve().parent / "feeds.json"
USER_AGENT = "Mozilla/5.0 (compatible; bfsi-daily/1.0)"
FEED_TIMEOUT_SECONDS = 15


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


def collect_news(feeds: list[dict], hours: int = 24) -> dict[str, list[str]]:
    """Fetch feeds, keep recent or undated items, and deduplicate by link or title."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    seen_links: set[str] = set()
    seen_titles: set[str] = set()
    results: dict[str, list[str]] = {}

    for feed_info in feeds:
        source_name = feed_info["name"]
        source_type = feed_info.get("type", "direct")
        label = f"{source_name} [{source_type}]" if source_type == "aggregator" else source_name
        url = feed_info["url"]
        results[label] = []

        try:
            parsed_feed = fetch_feed(url)
        except Exception as exc:
            print(f"[Warning] Skipping {source_name} ({url}): {exc}")
            continue

        if getattr(parsed_feed, "bozo", False) and not parsed_feed.entries:
            print(f"[Warning] Skipping {source_name} ({url}): {parsed_feed.bozo_exception}")
            continue

        for entry in parsed_feed.entries:
            published_at = get_entry_datetime(entry)
            if published_at is not None and published_at < cutoff:
                continue

            title = (entry.get("title") or "").strip()
            link = (entry.get("link") or "").strip()
            title_key = title.lower()

            if (link and link in seen_links) or (title_key and title_key in seen_titles):
                continue

            if link:
                seen_links.add(link)
            if title_key:
                seen_titles.add(title_key)

            results[label].append(title or "(No title)")

    return results


def main() -> None:
    """Load feeds, collect recent unique articles, and print counts and titles per source."""
    feeds = load_feeds(FEEDS_FILE)
    news_by_source = collect_news(feeds)

    for source_name, titles in news_by_source.items():
        print(f"\n{source_name}: {len(titles)} item(s)")
        for title in titles:
            print(f"  - {title}")


if __name__ == "__main__":
    main()
