"""Minimal EURI example matching the requested integration. Run from project root."""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
if not os.environ.get("EURI_API_KEY"):
    raise SystemExit("Add EURI_API_KEY to .env before running this live example.")

with OpenAI(
    api_key=os.environ["EURI_API_KEY"],
    base_url="https://api.euron.one/api/v1/euri",
    timeout=45,
) as client:
    resp = client.chat.completions.create(
        model="gemini-3.5-flash-lite",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a haiku about the ocean."},
        ],
        temperature=0.7,
    )
    print(resp.choices[0].message.content)
