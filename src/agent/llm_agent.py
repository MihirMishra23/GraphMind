from __future__ import annotations

"""Agent that queries an LLM to decide the next action and manages its own memory."""
from pathlib import Path
from typing import List, Optional, Sequence

from llm import LLM
from memory.entity_extractor import CandidateFact
from memory.schema import WorldKG

from .base import BaseAgent


DEFAULT_SYSTEM_PROMPT = """You are controlling a player in a parser-based text adventure game.
You receive the latest observation from the game and a short history of recent actions.
Reply with a single valid game command (e.g., 'open door', 'get lamp', 'north').
Do not include explanations or quotes, only the command text."""


class LLMAgent(BaseAgent):
    def __init__(
        self,
        llm: LLM,
        memory_mode: str = "none",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        history_horizon: int = 5,
        max_tokens: int = 32,
        temperature: float = 0.5,
        actions_key: str = "valid_actions",
        extraction_max_tokens: int = 256,
        extraction_temperature: float = 0.0,
        extraction_mode: str = "llm",
    ) -> None:
        super().__init__(use_memory=memory_mode != "none")
        self.llm = llm
        self.memory_mode = memory_mode
        self.system_prompt = system_prompt
        self.history_horizon = history_horizon
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.actions_key = actions_key
        self.extraction_max_tokens = extraction_max_tokens
        self.extraction_temperature = extraction_temperature
        self.extraction_mode = extraction_mode
        self._last_action: Optional[str] = None

    def act(self, observation: str, action_candidates: List[str]) -> Optional[str]:
        kg_text = self._build_kg_context(observation)
        proposed = self.propose_action(observation, kg_text, self._recent_history)
        action = self._choose_from_candidates(proposed, action_candidates)
        self._last_action = action
        return action or "look"

    def propose_action(self, obs: str, kg_text: str, recent_history: list[str]) -> str:
        history_block = (
            "\n".join(recent_history[-self.history_horizon :]) if recent_history else "None"
        )
        kg_block = kg_text.strip() if kg_text.strip() else "None"
        prompt = (
            f"{self.system_prompt}\n\n"
            f"Recent history:\n{history_block}\n\n"
            f"Known context:\n{kg_block}\n\n"
            f"Latest observation:\n{obs}\n\n"
            f"Next action (one command):"
        )
        completion = self.llm.generate(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["\n"],
        )
        return completion.strip()

    def extract_entities_and_relations(
        self, prev_obs: str | None, action: str | None, obs: str
    ) -> list[CandidateFact]:
        """Use heuristic extraction unless disabled or overridden."""
        if not self.use_memory:
            return []
        return super().extract_entities_and_relations(prev_obs, action, obs)

    def export_memory(
        self,
        dot_path: Path,
        include_inactive: bool = False,
        png_path: Optional[Path] = None,
    ) -> None:
        """Placeholder for future visualization with the new KG-based memory."""
        return None

    def _choose_from_candidates(
        self, completion: str, action_candidates: Sequence[str]
    ) -> str:
        lines = completion.strip().splitlines()
        cleaned = lines[0] if lines else ""
        cleaned_lower = cleaned.lower()
        # Exact match
        for cand in action_candidates:
            if cleaned_lower == cand.lower():
                return cand
        # Prefix/substring match
        for cand in action_candidates:
            if cleaned_lower.startswith(cand.lower()) or cand.lower().startswith(
                cleaned_lower
            ):
                return cand
        # Fallback to first candidate
        return action_candidates[0]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _build_kg_context(self, observation: str) -> str:
        """
        Render a small, relevant subgraph around the player/context as text for
        the prompt. Returns empty string when memory is disabled.
        """
        if not self.use_memory or not self.retriever or not self.world_kg:
            return ""
        step = self._last_step if self._last_step is not None else 0
        subgraph: WorldKG = self.retriever.get_relevant_subgraph(observation, step=step, radius=2)
        return subgraph.to_text_summary()
