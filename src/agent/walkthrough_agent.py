from __future__ import annotations

"""Simple agent that replays the Jericho-provided walkthrough if available."""
from typing import List, Optional

from .base import BaseAgent


class WalkthroughAgent(BaseAgent):
    def __init__(self, memory: Optional[object] = None) -> None:
        super().__init__(memory=memory)
        self._actions: List[str] = []
        self._cursor: int = 0

    def reset(self, env: object) -> None:
        super().reset(env)
        # Jericho environments expose get_walkthrough for deterministic trajectories.
        walkthrough = getattr(env, "get_walkthrough", lambda: [])()
        self._actions = list(walkthrough) if walkthrough is not None else []
        self._cursor = 0

    def act(self, observation: str, info: Optional[dict] = None) -> Optional[str]:
        if self._cursor < len(self._actions):
            action = self._actions[self._cursor]
            self._cursor += 1
            return action
        # Default fallback when walkthrough ends.
        return "look"
