from __future__ import annotations

"""Base classes for agent implementations."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


class BaseAgent(ABC):
    """Base interface for all agents that own their memory."""

    def __init__(self, memory: Optional[Any] = None) -> None:
        self.memory = memory
        self.env = None

    def reset(self, env: Any) -> None:
        """Attach environment and reset memory if provided."""
        self.env = env
        if hasattr(self.memory, "reset"):
            self.memory.reset()

    def observe(
        self,
        turn_id: int,
        action: str,
        observation: str,
        reward: float,
        info: Optional[dict] = None,
    ) -> None:
        """Hook for agents to record a completed step; default forwards to memory if available."""
        if hasattr(self.memory, "observe"):
            self.memory.observe(turn_id, action, observation, reward, info)

    def export_memory(self, dot_path: Path, include_inactive: bool = False, png_path: Optional[Path] = None) -> None:
        """Optional visualization hook; overridden by agents that own graph memory."""
        return None

    @abstractmethod
    def act(self, observation: str, action_candidates: list[str]) -> Optional[str]:
        """Return the next action string for the environment."""
        raise NotImplementedError
