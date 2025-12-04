from __future__ import annotations

"""OpenAI Chat Completions client wrapper."""
import os
from typing import Iterable, Optional

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - dependency not installed in all envs
    OpenAI = None  # type: ignore
    _import_error = exc
else:
    _import_error = None

from .base import LLM


class OpenAILLM(LLM):
    """
    Minimal wrapper for OpenAI chat models (defaults to gpt-4o-mini).

    Expects OPENAI_API_KEY to be set in the environment, or an api_key provided.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        **kwargs,
    ) -> None:
        if OpenAI is None:
            raise _import_error or RuntimeError(
                "openai package is required for OpenAILLM but is not installed"
            )
        self.model = model
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY must be set for OpenAILLM")
        self.client = OpenAI(api_key=key, **kwargs)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        stop: Optional[Iterable[str]] = None,
    ) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
            stop=list(stop) if stop else None,
        )
        return (completion.choices[0].message.content or "").strip()
