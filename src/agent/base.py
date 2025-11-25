from __future__ import annotations

"""Base classes for agent implementations."""
from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseAgent(ABC):
    """Base interface for all agents.

    Agents can optionally hold a reference to memory; runners are responsible
    for wiring environment and memory before stepping.
    """

    def __init__(self, memory: Optional[Any] = None) -> None:
        self.memory = memory
        self.env = None

    def reset(self, env: Any) -> None:
        """Attach environment and reset memory if provided."""
        self.env = env
        if hasattr(self.memory, "reset"):
            self.memory.reset()

    @abstractmethod
    def act(self, observation: str, info: Optional[dict] = None) -> Optional[str]:
        """Return the next action string for the environment."""
        raise NotImplementedError
