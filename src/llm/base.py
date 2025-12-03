from __future__ import annotations

"""Abstract interfaces for LLM clients."""
from abc import ABC, abstractmethod
from typing import Iterable, Optional


class LLM(ABC):
    """Base contract for raw text-generation models."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        stop: Optional[Iterable[str]] = None,
    ) -> str:
        """Generate a completion for the given prompt."""
        raise NotImplementedError
