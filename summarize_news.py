from datetime import date
import json
import os
from pathlib import Path
import re
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

from collect_news import collect_news


OUTPUT_FILE = Path(__file__).resolve().parent / "news.json"
MAX_RETRIES = 3
INITIAL_RETRY_WAIT_SECONDS = 5
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
ALLOWED_CATEGORIES = (
    "Life",
    "General",
    "Health",
    "Regulatory",
    "Banking",
    "Fintech",
    "Markets",
)

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "stories": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "INTEGER"},
                    "headline": {"type": "STRING"},
                    "summary": {"type": "STRING"},
                    "category": {"type": "STRING", "enum": list(ALLOWED_CATEGORIES)},
                    "why_it_matters": {"type": "STRING"},
                },
                "required": ["id", "headline", "summary", "category", "why_it_matters"],
            },
        }
    },
    "required": ["stories"],
}

PROMPT_TEMPLATE = """You are curating a daily news digest for insurance and BFSI (banking, financial services, and insurance) students and professionals in India.

From the news items below, select 8 stories following these strict rules:

SELECTION MIX:
- Choose 8 stories in total.
- Include at least 4 stories about insurance (life, general, health, IRDAI, reinsurance, bancassurance, or insurtech). If fewer than 4 insurance stories exist, fill the remaining slots with the best other BFSI stories.
- Include at least 1 but at most 2 stories about RBI, banking, or fintech combined.
- Ignore gold/silver prices, stock tips, celebrity news, and unrelated general company news.
- If multiple items cover the same underlying story, pick only ONE item.
- Order stories by importance: regulatory and IRDAI news first, then other insurance, then RBI/banking/fintech.

WRITING RULES — apply every rule to every field:
1. Write every summary entirely in your own words. Do not copy any run of more than 4 consecutive words from the source text, except proper names, numbers, official titles, and exact product or scheme names.
2. Keep every fact exactly as given in the source; do not round, estimate, or add context.
3. Never describe a proposal or draft circular as final; use "proposed" or "draft" when the source text says so.
4. Do not refer to any person with he, she, his, or her unless the source text itself uses that pronoun; use the person's name or job title instead.
5. Do not use relative time phrases such as "today", "yesterday", "last year", "on Monday", "ahead of Diwali", "last quarter", or similar; use only dates explicitly stated in the text.
6. If the source does not state whether a figure is month-on-month or year-on-year, do not specify either.
7. Use ONLY facts present in that item's title and snippet; never invent numbers, dates, or claims.
8. When the text gives a period or year (such as "April–September 2026" or "FY27"), include it in the summary.
9. When the text names the publisher of a report or data (for example "a BCG-IIA report" or "RBI data"), name them in the summary.

OUTPUT FIELDS (return strict JSON matching the schema):
- id: exact integer id from the input items
- headline: plain, clear headline (max 15 words)
- summary: exactly 2 sentences with key facts first; name products and schemes by their exact names
- category: one of Life, General, Health, Regulatory, Banking, Fintech, Markets
- why_it_matters: one sentence naming the one concept or consequence that genuinely applies to this story — name it in plain words (for example a regulatory mechanism, a type of market participant, a customer protection rule, or a financial consequence for insurers or policyholders). If no technical concept applies, state the practical effect for insurers, banks, or customers instead. Do not force a term that is not relevant. Do not use filler words like "crucial" or "valuable". Do not start with "Students".

Input items (JSON):
{items_json}
"""

REWRITE_PROMPT_TEMPLATE = """The following story summaries were found to repeat five or more consecutive words from their source text. Rewrite only the summary field for each story in entirely different words while keeping every fact exactly as given.

Return strict JSON matching this structure: {{"stories": [{{"id": <integer>, "summary": "<rewritten 2-sentence summary>"}}]}}

Stories to rewrite (each includes the original source title and snippet for reference, and the summary to fix):
{stories_json}
"""


def load_config() -> tuple[str, str]:
    """Load required Gemini settings from .env / environment variables."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "").strip()

    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in environment or .env file.")
    if not model:
        raise ValueError("Missing GEMINI_MODEL in environment or .env file.")

    return api_key, model


def call_gemini_with_retry(client: genai.Client, model: str, prompt: str) -> str:
    """Send one request to Gemini, retrying up to 3 times on transient errors like 429/503."""
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
        temperature=0.2,
    )

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            if not response.text:
                raise RuntimeError("Gemini returned an empty response.")
            return response.text
        except APIError as exc:
            is_retryable = getattr(exc, "code", None) in RETRYABLE_STATUS_CODES
            if not is_retryable or attempt >= MAX_RETRIES:
                raise
            wait_seconds = INITIAL_RETRY_WAIT_SECONDS * (2**attempt)
            print(
                f"[Warning] Gemini API error ({exc.code}). "
                f"Retrying in {wait_seconds}s (attempt {attempt + 1}/{MAX_RETRIES})...",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)
        except (TimeoutError, OSError) as exc:
            if attempt >= MAX_RETRIES:
                raise
            wait_seconds = INITIAL_RETRY_WAIT_SECONDS * (2**attempt)
            print(
                f"[Warning] Network error ({exc}). "
                f"Retrying in {wait_seconds}s (attempt {attempt + 1}/{MAX_RETRIES})...",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)

    raise RuntimeError("Exhausted Gemini retries.")


def call_gemini_plain(client: genai.Client, model: str, prompt: str) -> str:
    """Send a plain (no schema) JSON request to Gemini without retry — used for the rewrite pass."""
    rewrite_schema = {
        "type": "OBJECT",
        "properties": {
            "stories": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "id": {"type": "INTEGER"},
                        "summary": {"type": "STRING"},
                    },
                    "required": ["id", "summary"],
                },
            }
        },
        "required": ["stories"],
    }
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=rewrite_schema,
        temperature=0.4,
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    if not response.text:
        raise RuntimeError("Gemini returned an empty response on rewrite pass.")
    return response.text


def _tokenise(text: str) -> list[str]:
    """Split text into lowercase word tokens, stripping punctuation."""
    return re.findall(r"[a-z]+", text.lower())


def _is_skip_token(token: str, raw_token: str) -> bool:
    """Return True for pure-number tokens or fully-capitalised name tokens (1-char exempt)."""
    if re.fullmatch(r"\d+", raw_token):
        return True
    if len(raw_token) > 1 and raw_token == raw_token.upper():
        return True
    return False


def find_verbatim_overlap(summary: str, source_title: str, source_snippet: str) -> bool:
    """Return True if the summary contains a run of 5+ consecutive non-exempt source words."""
    source_text = f"{source_title} {source_snippet}"
    source_tokens = _tokenise(source_text)
    summary_tokens = _tokenise(summary)

    # Build set of all 5-grams from the source, excluding runs that are entirely skip-tokens.
    source_raw = re.findall(r"\S+", source_text.lower())
    summary_raw = re.findall(r"\S+", summary.lower())

    def meaningful_ngrams(raw_words: list[str], n: int = 5) -> set[tuple[str, ...]]:
        grams: set[tuple[str, ...]] = set()
        for i in range(len(raw_words) - n + 1):
            window = raw_words[i : i + n]
            # Skip the window if every token is a number or an all-caps proper name
            stripped = [re.sub(r"[^a-z0-9]", "", w) for w in window]
            if all(_is_skip_token(s, w) for s, w in zip(stripped, window)):
                continue
            grams.add(tuple(re.sub(r"[^a-z]", "", w) for w in window))
        return grams

    source_grams = meaningful_ngrams(source_raw)
    if not source_grams:
        return False

    for i in range(len(summary_raw) - 4):
        window = summary_raw[i : i + 5]
        key = tuple(re.sub(r"[^a-z]", "", w) for w in window)
        if key in source_grams:
            return True
    return False


def check_overlap(
    stories: list[dict], items_by_id: dict[int, dict]
) -> list[int]:
    """Return ids of stories whose summary has a 5-word verbatim overlap with source text."""
    flagged: list[int] = []
    for story in stories:
        original = items_by_id.get(story["id"])
        if original is None:
            continue
        if find_verbatim_overlap(
            story["summary"], original["title"], original["snippet"]
        ):
            flagged.append(story["id"])
    return flagged


def apply_rewrites(
    stories: list[dict],
    rewrite_raw: str,
) -> None:
    """Merge rewritten summaries back into stories in-place."""
    try:
        payload = json.loads(rewrite_raw)
        rewrites = {r["id"]: r["summary"] for r in payload.get("stories", []) if isinstance(r, dict)}
    except Exception:
        return
    for story in stories:
        if story["id"] in rewrites:
            story["summary"] = str(rewrites[story["id"]]).strip()


def build_stories(raw_json: str, items_by_id: dict[int, dict]) -> list[dict]:
    """Parse Gemini JSON output and attach verified source and link from original items."""
    payload = json.loads(raw_json)
    raw_stories = payload.get("stories", [])
    if not isinstance(raw_stories, list):
        raise ValueError("Model response JSON must contain a 'stories' list.")

    verified_stories: list[dict] = []
    seen_ids: set[int] = set()

    for story in raw_stories:
        if not isinstance(story, dict):
            continue

        story_id = story.get("id")
        if not isinstance(story_id, int) or story_id in seen_ids:
            continue

        original = items_by_id.get(story_id)
        if original is None:
            continue

        category = story.get("category", "")
        if category not in ALLOWED_CATEGORIES:
            continue

        seen_ids.add(story_id)
        verified_stories.append(
            {
                "id": story_id,
                "headline": str(story.get("headline", "")).strip(),
                "summary": str(story.get("summary", "")).strip(),
                "category": category,
                "why_it_matters": str(story.get("why_it_matters", "")).strip(),
                "source": original["source"],
                "link": original["link"],
            }
        )

    return verified_stories


def save_digest(output_path: Path, stories: list[dict]) -> None:
    """Write the daily digest to news.json safely via a temporary file."""
    digest = {
        "date": date.today().isoformat(),
        "stories": stories,
    }
    temp_path = output_path.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(digest, file, indent=2, ensure_ascii=False)
    temp_path.replace(output_path)


def main() -> None:
    """Collect news items, summarize top 8 stories with Gemini, and save news.json."""
    try:
        api_key, model = load_config()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)

    _, items = collect_news()
    if not items:
        print("Error: No news items were collected; news.json was not updated.", file=sys.stderr)
        sys.exit(1)

    items_by_id = {item["id"]: item for item in items}
    prompt_items = [
        {"id": item["id"], "title": item["title"], "snippet": item["snippet"]}
        for item in items
    ]
    prompt = PROMPT_TEMPLATE.format(
        items_json=json.dumps(prompt_items, ensure_ascii=False)
    )

    client = genai.Client(api_key=api_key)

    # --- Request 1: main summarisation ---
    try:
        raw_response = call_gemini_with_retry(client, model, prompt)
        stories = build_stories(raw_response, items_by_id)
    except Exception as exc:
        print(f"Error generating summary: {exc}", file=sys.stderr)
        sys.exit(1)

    if not stories:
        print("Error: Model returned no valid stories; news.json was not updated.", file=sys.stderr)
        sys.exit(1)

    # --- Verbatim-overlap check ---
    flagged_ids = check_overlap(stories, items_by_id)

    if flagged_ids:
        # --- Request 2: rewrite pass (at most one follow-up request total) ---
        stories_to_rewrite = [
            {
                "id": s["id"],
                "source_title": items_by_id[s["id"]]["title"],
                "source_snippet": items_by_id[s["id"]]["snippet"],
                "current_summary": s["summary"],
            }
            for s in stories
            if s["id"] in flagged_ids
        ]
        rewrite_prompt = REWRITE_PROMPT_TEMPLATE.format(
            stories_json=json.dumps(stories_to_rewrite, ensure_ascii=False)
        )
        try:
            rewrite_raw = call_gemini_plain(client, model, rewrite_prompt)
            apply_rewrites(stories, rewrite_raw)
        except Exception as exc:
            print(f"[Warning] Rewrite pass failed ({exc}); keeping original summaries.", file=sys.stderr)

        # Check again and warn for any that still overlap
        still_flagged = check_overlap(stories, items_by_id)
        for sid in still_flagged:
            print(
                f"[Warning] Story id={sid} still contains a verbatim overlap after rewrite; keeping as-is.",
                file=sys.stderr,
            )

    save_digest(OUTPUT_FILE, stories)
    print(f"Saved {len(stories)} stories for {date.today().isoformat()} to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
