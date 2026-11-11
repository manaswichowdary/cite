"""Settings for Cite. Keys come from the environment or a local .env file and are never printed."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    # repr=False keeps keys out of logs and tracebacks.
    openai_api_key: str = field(default="", repr=False)
    tavily_api_key: str = field(default="", repr=False)
    model: str = "gpt-4o-mini"
    max_usd_per_question: float = 0.05
    max_steps: int = 8
    daily_cap_usd: float = 0.50     # the web server refuses new questions after this much spend

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key.strip())

    @property
    def has_tavily(self) -> bool:
        return bool(self.tavily_api_key.strip())


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Read settings. Pass `env` in tests; otherwise load .env and use the real environment."""
    if env is None:
        load_dotenv()
        env = os.environ
    return Settings(
        openai_api_key=env.get("OPENAI_API_KEY", ""),
        tavily_api_key=env.get("TAVILY_API_KEY", ""),
        model=env.get("CITE_MODEL", "gpt-4o-mini"),
        max_usd_per_question=float(env.get("CITE_MAX_USD_PER_QUESTION", "0.05")),
        max_steps=int(env.get("CITE_MAX_STEPS", "8")),
        daily_cap_usd=float(env.get("CITE_DAILY_CAP_USD", "0.50")),
    )
