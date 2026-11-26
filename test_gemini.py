import os
import sys
import traceback

from dotenv import load_dotenv
from google import genai
from google.genai import types


def main() -> None:
    """Send a minimal test request to Gemini using .env settings and lowest thinking level."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "").strip()

    if not api_key or not model:
        print("Error: GEMINI_API_KEY and GEMINI_MODEL must be set in .env", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=model,
            contents="Reply with exactly: working",
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="LOW")
            ),
        )
        print(response.text)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
