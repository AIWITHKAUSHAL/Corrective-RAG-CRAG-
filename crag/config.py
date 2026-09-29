"""Server-side configuration. API keys never appear in graph state or responses."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default="", repr=False)
    base_url: str = "https://api.euron.one/api/v1/euri"
    model: str = "gemini-3.5-flash-lite"
    tavily_key: str = field(default="", repr=False)

    @classmethod
    def from_env(cls):
        load_dotenv(ROOT / ".env", override=False)
        return cls(
            api_key=os.getenv("EURI_API_KEY", "").strip(),
            base_url=os.getenv("EURI_BASE_URL", cls.base_url).strip(),
            model=os.getenv("EURI_MODEL", cls.model).strip(),
            tavily_key=os.getenv("TAVILY_API_KEY", "").strip(),
        )
