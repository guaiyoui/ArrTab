"""Small OpenAI-compatible LLM adapter with strict JSON parsing."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMConfig:
    model: str
    base_url: str | None
    api_key: str
    max_tokens: int = 1024

    @classmethod
    def from_env(cls, model: str | None = None, base_url: str | None = None) -> LLMConfig:
        from dotenv import load_dotenv

        load_dotenv()
        api_key = (
            os.getenv("AGENTICTQA_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("NEBIUS_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "Missing API key. Set AGENTICTQA_API_KEY (or OPENAI_API_KEY/NEBIUS_API_KEY)."
            )
        return cls(
            model=model or os.getenv("AGENTICTQA_MODEL") or "Qwen/Qwen3-30B-A3B-Instruct-2507",
            base_url=base_url
            or os.getenv("AGENTICTQA_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.studio.nebius.com/v1/",
            api_key=api_key,
        )


class LLM:
    def __init__(self, config: LLMConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only before installation
            raise RuntimeError("Install the package first: pip install -e .") from exc
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.calls = 0

    def json(self, *, system: str, prompt: str) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=0,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        self.calls += 1
        content = response.choices[0].message.content or ""
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
        try:
            value = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model returned invalid JSON: {content[:300]}") from exc
        if not isinstance(value, dict):
            raise TypeError("Model response must be a JSON object")
        return value
