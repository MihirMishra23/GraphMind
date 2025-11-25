"""Placeholder memory implementation that records steps without retrieval logic."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class NullMemory:
    """No-op memory placeholder so runner wiring remains stable."""

    def __init__(self) -> None:
        self.trace: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.trace.clear()

    def observe(self, turn_id: int, action: str, observation: str, reward: float, info: Optional[dict] = None) -> None:
        self.trace.append(
            {
                "turn_id": turn_id,
                "action": action,
                "observation": observation,
                "reward": reward,
                "info": info or {},
            }
        )

    def retrieve(self, query: Optional[str] = None) -> List[Dict[str, Any]]:
        return list(self.trace)
