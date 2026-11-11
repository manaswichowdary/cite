"""Quick setup check. Prints yes or no for each key, never the key itself."""

from __future__ import annotations

import sys

from .config import load_settings


def main() -> int:
    s = load_settings()
    print(f"OpenAI key set: {'yes' if s.has_openai else 'no'}")
    print(f"Tavily key set: {'yes' if s.has_tavily else 'no'}")
    print(f"Model: {s.model}")
    print(f"Spending cap per question: ${s.max_usd_per_question:.2f}")
    print(f"Max agent steps: {s.max_steps}")
    return 0 if (s.has_openai and s.has_tavily) else 1


if __name__ == "__main__":
    sys.exit(main())
