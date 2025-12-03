from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from llm import LLM

if TYPE_CHECKING:
    from memory.entity_extractor import CandidateFact


class BasePolicy(ABC):
    """High-level interface for the agent + memory system."""

    @abstractmethod
    def propose_action(self, obs: str, kg_text: str, recent_history: list[str]) -> str:
        """Return next action as a text command."""

    @abstractmethod
    def extract_entities_and_relations(
        self, prev_obs: str | None, action: str | None, obs: str
    ) -> list["CandidateFact"]:
        """
        Optional: Alternative LLM-based extractor. For now, unused.
        """


DEFAULT_POLICY_PROMPT = """You are controlling a player in a text adventure game.
Decide the single best next command to progress.
Return only the command text without quotes or commentary."""


class LLMPolicy(BasePolicy):
    """
    Simple LLM-driven policy that proposes the next action given the current
    observation, a textual KG summary, and recent history.
    """

    def __init__(
        self,
        llm: LLM,
        system_prompt: str = DEFAULT_POLICY_PROMPT,
        max_tokens: int = 32,
        temperature: float = 0.5,
        history_horizon: int = 5,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.history_horizon = history_horizon

    def propose_action(self, obs: str, kg_text: str, recent_history: list[str]) -> str:
        history_block = (
            "\n".join(recent_history[-self.history_horizon :])
            if recent_history
            else "None"
        )
        kg_block = kg_text.strip() if kg_text.strip() else "None"
        prompt = (
            f"{self.system_prompt}\n\n"
            f"Recent history:\n{history_block}\n\n"
            f"Known context:\n{kg_block}\n\n"
            f"Latest observation:\n{obs}\n\n"
            f"Next action:"
        )
        completion = self.llm.generate(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["\n"],
        )
        action = completion.strip()
        return action or "look"

    def extract_entities_and_relations(
        self, prev_obs: str | None, action: str | None, obs: str
    ) -> list["CandidateFact"]:
        # Placeholder: delegate to a dedicated extractor when available.
        return []


class DummyPolicy(BasePolicy):
    def propose_action(self, obs: str, kg_text: str, recent_history: list[str]) -> str:
        """
        Placeholder policy: returns a hardcoded or random action.
        """
        return "look"

    def extract_entities_and_relations(self, prev_obs, action, obs):
        return []
