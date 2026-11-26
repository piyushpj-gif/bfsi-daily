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

PROMPT_TEMPLATE = """You are curating a daily news digest for insurance and BFSI (banking, financial services and insurance) students and professionals in India. Readers include industry experts, so accuracy matters more than anything else.

From the news items below, select 8 stories following these rules.

SELECTION:
- Choose exactly 8 stories.
- Include at least 4 stories about insurance (life, general, health, IRDAI, reinsurance, bancassurance or insurtech). If fewer than 4 insurance stories exist, fill the remaining slots with the best other BFSI stories.
- Include at least 1 and at most 2 stories about RBI, banking or fintech combined.
- Skip routine operations such as RBI auctions, T-bill results, money market operations and bulletins, unless the item announces a significant policy or regulatory change.
- Ignore gold and silver prices, stock tips, celebrity news and unrelated company news.
- If several items cover the same underlying story, pick only ONE of them.
- Order the stories by importance: regulatory and IRDAI news first, then other insurance, then RBI, banking and fintech.
- Skip advertising campaigns, awards and sponsorships unless they have a clear regulatory or market impact.

ACCURACY RULES (most important):
1. Use ONLY facts present in that item's title and snippet. Never invent or guess numbers, dates, names or claims.
2. Keep every fact exactly as given. Do not round, estimate or add context from your own knowledge.
3. Keep standard industry terms and job titles exactly as the source uses them (for example "Appointed Actuary", "cyber insurance", "property insurance", "study groups", "credit card"). Never replace a standard term with an unusual synonym.
4. Never describe a proposal or draft as final. Use "proposed" or "draft" when the text says so.
5. Do not say whether an appointment, launch or change needs regulatory approval unless the text says so, and apply that condition only to the person or item the text names.
6. If the text does not say whether a figure is month-on-month or year-on-year, do not state either.
7. Include the period or year when the text gives one, and name the publisher of a report or data when the text names one (for example "a BCG-IIA report" or "RBI data").
8. Do not refer to a person with he, she, his or her unless the text itself does. Use the name or job title.
9. Do not use relative time phrases such as today, yesterday, last year, last quarter, on Monday or ahead of a festival. Use only dates stated in the text.

WORDING RULES:
- Write summaries in your own sentence structure so they are not a copy of the source. Do not copy a run of more than 8 consecutive words from the source text, except names, numbers, titles and product names.
- Reorder and condense instead of swapping in unusual synonyms. If a rewrite would change the meaning or sound unnatural, keep the source's standard wording for that term.
- Use plain language a first-year student can follow.

HEADLINE-ONLY ITEMS:
- Some items have an empty or repeated snippet. For those, use only the facts in the title. Write two short sentences that restate the title's facts. Do not add purposes, goals, impacts, locations or background that the title does not state. Use at most 2 such items among the 8 stories.
- If the text says "according to a report" or names a person or body as the source of a figure, keep that attribution in both the headline and the summary (for example "reportedly").
- Do not state who organised or ran an event unless the text says so.

CATEGORY GUIDE:
- Life: life insurance, pensions and annuities.
- General: non-life insurance such as motor, property, travel, crop, cyber, and job-loss or income-protection cover.
- Health: health insurance only.
- Regulatory: rules, committees, circulars or actions by IRDAI, RBI, SEBI or the government.
- Banking and Fintech: banks, payments and fintech.
- Markets: funding, ownership, investments and leadership news about firms when the line of business is not stated in the text.

CONSEQUENCE WORDING:
- In why_it_matters, describe possible effects with cautious words such as can, may or could. Never present a prediction as a fact.
- Do not start with the word "When". Vary the sentence openings across stories.
- Name an industry concept (for example net interest margin or acquisition cost) only if the story's own facts involve it. Never introduce a technical term that the title and snippet do not support.
- One sentence of 15 to 30 words, with a point specific to this story rather than a general statement.

OUTPUT FIELDS (return strict JSON matching the schema):
- id: exact integer id from the input items
- headline: clear headline using standard terms, at most 15 words
- summary: exactly 2 sentences, key facts first, product and scheme names exact
- category: one of Life, General, Health, Regulatory, Banking, Fintech, Markets
- why_it_matters: ONE complete sentence of 15 to 30 words with a subject and a verb. Explain what this specific story means for insurers, banks, policyholders or customers. Where one clearly applies, name the relevant industry concept (for example appointed actuary, claims settlement, solvency, underwriting cycle, loss ratio, bancassurance, or the difference between a draft and a final regulation). Never write a label or fragment such as "A financial consequence for insurers" or "A type of market participant". An example of the style, about a different topic: "When a regulator moves from draft to final rules, insurers must change their processes by the effective date, so the difference matters for compliance planning." Do not use filler words like crucial or valuable. Do not start with Students.

Input items (JSON):
{items_json}
"""


REWRITE_PROMPT_TEMPLATE = """The following story summaries repeat eight or more consecutive words from their source text. Rewrite only the summary field for each story by changing the sentence structure and the order of information, while keeping every fact exactly as given and keeping standard industry terms and job titles unchanged. Do not replace standard terms with unusual synonyms, and do not add facts.

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
    """Return True if the summary contains a run of 8+ consecutive non-exempt source words."""
    source_text = f"{source_title} {source_snippet}"
    source_tokens = _tokenise(source_text)
    summary_tokens = _tokenise(summary)

    # Build set of all 8-grams from the source, excluding runs that are entirely skip-tokens.
    source_raw = re.findall(r"\S+", source_text.lower())
    summary_raw = re.findall(r"\S+", summary.lower())

    def meaningful_ngrams(raw_words: list[str], n: int = 8) -> set[tuple[str, ...]]:
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

    for i in range(len(summary_raw) - 7):
        window = summary_raw[i : i + 8]
        key = tuple(re.sub(r"[^a-z]", "", w) for w in window)
        if key in source_grams:
            return True
    return False


def check_overlap(
    stories: list[dict], items_by_id: dict[int, dict]
) -> list[int]:
    """Return ids of stories whose summary has an 8-word verbatim overlap with source text."""
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
