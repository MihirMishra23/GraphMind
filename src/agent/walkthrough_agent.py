from __future__ import annotations

"""Simple agent that replays the Jericho-provided walkthrough if available."""
from typing import List, Optional

from .base import BaseAgent


class WalkthroughAgent(BaseAgent):
    def __init__(self, walkthrough: List, use_memory: bool = True) -> None:
        super().__init__(use_memory=use_memory)
        self._actions: List[str] = []
        self._cursor: int = 0
        self.walkthrough = iter(walkthrough)
        self._actions = list(walkthrough) if walkthrough is not None else []
        self._cursor = 0

    def reset(self, env: object) -> None:
        # Jericho environments expose get_walkthrough for deterministic trajectories.
        walkthrough = getattr(env, "get_walkthrough", lambda: [])()
        self._actions = list(walkthrough) if walkthrough is not None else []
        self._cursor = 0

    def act(self, observation: str, action_candidates: List[str]) -> Optional[str]:
        self._last_action_candidates = list(action_candidates)
        if self._cursor < len(self._actions):
            action = self._actions[self._cursor]
            self._cursor += 1
            return action
        # Default fallback when walkthrough ends.
        return "look"
