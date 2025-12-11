from __future__ import annotations

"""Base classes for agent implementations."""
from abc import ABC, abstractmethod
from typing import Any, Optional
import logging


class BaseAgent(ABC):
    """Minimal agent interface."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.env = None
        self._recent_steps: list[dict[str, str]] = []
        self._last_step: Optional[int] = None

    def reset(self, env: Any) -> None:
        """Attach environment and clear recent history."""
        self.env = env
        self._recent_steps.clear()
        self._last_step = None

    @abstractmethod
    def observe(self, observation: str) -> None:
        """Record a completed step for simple history heuristics."""
        raise NotImplementedError

    @abstractmethod
    def act(self, observation: str, action_candidates: list[str]) -> Optional[str]:
        """Return the next action string for the environment."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _get_recent_history_lines(self, horizon: int) -> list[str]:
        """
        Return structured recent history lines up to the provided horizon,
        alternating Action/Observation entries.
        """
        lines: list[str] = []
        for step in self._recent_steps[-horizon:]:
            lines.append(f"Action: {step['action']}")
            lines.append(f"Observation: {step['observation']}")
        return lines
