from __future__ import annotations

"""Abstract interface for LLM clients."""
from abc import ABC, abstractmethod
from typing import Iterable, Optional


class LLM(ABC):
    """Base contract for text-generation models."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.7,
        stop: Optional[Iterable[str]] = None,
    ) -> str:
        """Generate a completion for the given prompt."""
        raise NotImplementedError
