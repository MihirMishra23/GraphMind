from __future__ import annotations

import json
from typing import Any, List

from llm import LLM


class StateEffectModel:
    """
    Lightweight wrapper around an LLM to propose world-state effects given
    an action, observation, and extracted facts.
    """

    def __init__(self, llm: LLM, max_tokens: int = 256) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    def propose_effects(self, action: str, observation: str, facts: List[dict[str, Any]]) -> list[dict[str, Any]]:
        prompt = self._build_prompt(action, observation, facts)
        completion = self.llm.generate(
            prompt, max_tokens=self.max_tokens, stop=["\n\n", "\nReturn"]
        )
        try:
            effects = json.loads(completion.strip())
        except Exception:
            return []
        if not isinstance(effects, list):
            return []
        return [e for e in effects if isinstance(e, dict)]

    def _build_prompt(self, action: str, observation: str, facts: List[dict[str, Any]]) -> str:
        facts_json = json.dumps(facts, ensure_ascii=False)
        return (
            "Decide how the world state changes after an action.\n"
            "Given the action, resulting observation, and extracted facts, return a JSON array of effect objects.\n"
            "Each effect object keys:\n"
            "  op: add_inventory | set_state | contains | connect | noop\n"
            "  target: string (entity name)\n"
            "  container: optional string (for contains)\n"
            "  state_updates: object (for set_state)\n"
            "  confidence: float 0-1\n"
            "Use empty array if no change. Keep responses concise and valid JSON.\n"
            f"Action: {action}\n"
            f"Observation: {observation}\n"
            f"Extracted facts (JSON): {facts_json}\n"
            "JSON array:"
        )


__all__ = ["StateEffectModel"]
