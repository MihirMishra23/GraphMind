from __future__ import annotations

"""Agent that queries an LLM to decide the next action."""
from typing import Dict, List, Optional, Sequence

from llm import LLM

from .base import BaseAgent


DEFAULT_SYSTEM_PROMPT = """You are controlling a player in a parser-based text adventure game.
You receive the latest observation from the game and a short history of recent actions.
Reply with a single valid game command (e.g., 'open door', 'get lamp', 'north').
If you are unsure, prefer exploratory but safe actions like 'look' or checking inventory with 'inventory'.
Do not include explanations or quotes, only the command text."""


class LLMAgent(BaseAgent):
    def __init__(
        self,
        llm: LLM,
        memory: Optional[object] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        history_horizon: int = 5,
        max_tokens: int = 32,
        temperature: float = 0.5,
        actions_key: str = "valid_actions",
    ) -> None:
        super().__init__(memory=memory)
        self.llm = llm
        self.system_prompt = system_prompt
        self.history_horizon = history_horizon
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.actions_key = actions_key
        self._recent: List[Dict[str, str]] = []

    def reset(self, env: object) -> None:
        super().reset(env)
        self._recent.clear()

    def act(self, observation: str, info: Optional[dict] = None) -> Optional[str]:
        candidates = self._extract_candidates(info)
        if not candidates:
            candidates = ["look"]

        prompt = self._build_prompt(observation, candidates)
        completion = self.llm.generate(
            prompt, max_tokens=self.max_tokens, temperature=self.temperature, stop=["\n"]
        )
        action = self._choose_from_candidates(completion, candidates)
        self._push_history(action, observation)
        return action or "look"

    def _build_prompt(self, observation: str, candidates: Sequence[str]) -> str:
        recent_lines = []
        for row in self._recent[-self.history_horizon :]:
            recent_lines.append(f"Action: {row['action']}")
            recent_lines.append(f"Observation: {row['observation']}")

        joined_history = "\n".join(recent_lines) if recent_lines else "None"
        options = "\n".join(f"- {a}" for a in candidates)
        prompt = (
            f"{self.system_prompt}\n\n"
            f"Recent history:\n{joined_history}\n\n"
            f"New observation:\n{observation}\n\n"
            f"Allowed actions (choose exactly one):\n{options}\n\n"
            f"Next action (copy exactly one of the allowed actions):"
        )
        return prompt

    def _push_history(self, action: str, observation: str) -> None:
        self._recent.append({"action": action, "observation": observation})

    def _extract_candidates(self, info: Optional[dict]) -> List[str]:
        if info is None:
            return []
        candidates = info.get(self.actions_key) or info.get("admissible_commands") or []
        return [c for c in candidates if isinstance(c, str)]

    def _choose_from_candidates(self, completion: str, candidates: Sequence[str]) -> str:
        lines = completion.strip().splitlines()
        cleaned = lines[0] if lines else ""
        cleaned_lower = cleaned.lower()
        # Exact match
        for cand in candidates:
            if cleaned_lower == cand.lower():
                return cand
        # Prefix/substring match
        for cand in candidates:
            if cleaned_lower.startswith(cand.lower()) or cand.lower().startswith(cleaned_lower):
                return cand
        # Fallback to first candidate
        return candidates[0]
