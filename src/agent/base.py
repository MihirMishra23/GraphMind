from __future__ import annotations

"""Base classes for agent implementations."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Sequence
import logging

from memory.entity_extractor import (
    CandidateFact,
    EntityRelationExtractor,
    NaiveEntityRelationExtractor,
)
from memory.kg_store import KGSnapshots
from memory.memory_manager import MemoryManager
from memory.retriever import KGRetriever
from memory.schema import WorldKG


class BaseAgent(ABC):
    """Base interface for all agents that own their memory."""

    def __init__(self, use_memory: bool = True) -> None:
        self.use_memory = use_memory
        self.logger = logging.getLogger(self.__class__.__name__)
        self.env = None
        self.world_kg: Optional[WorldKG] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.entity_extractor: Optional[EntityRelationExtractor] = None
        self.kg_snapshots: Optional[KGSnapshots] = None
        self.retriever: Optional[KGRetriever] = None
        self._recent_steps: list[dict[str, str]] = []
        self._last_step: Optional[int] = None
        self._prev_obs: Optional[str] = None
        self._last_obs_node_id: Optional[str] = None
        if self.use_memory:
            self._init_memory_backend()

    def reset(self, env: Any) -> None:
        """Attach environment and reset memory if provided."""
        self.env = env
        self._recent_steps.clear()
        self._last_step = None
        self._prev_obs = None
        if self.use_memory:
            self._init_memory_backend()
        self._last_obs_node_id = None

    def observe(
        self,
        turn_id: int,
        action: str,
        observation: str,
        reward: float,
        info: Optional[dict] = None,
    ) -> None:
        """
        Hook for agents to record a completed step; updates the structured memory
        using the active extractor + memory manager when enabled.
        """
        self._last_step = turn_id
        if action:
            self._recent_steps.append({"action": action, "observation": observation})

        if self.use_memory and self.memory_manager and self.world_kg:
            curr_obs_node = self._record_observation_transition(action, observation, turn_id)
            facts = self.extract_entities_and_relations(
                self._prev_obs, action, observation
            )
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Extraction produced %d candidate facts at step %s", len(facts), turn_id
                )
            self.memory_manager.decide_and_apply(facts, step=turn_id, observation_node_id=curr_obs_node)
            if self.kg_snapshots:
                self.kg_snapshots.store_snapshot(turn_id, self.world_kg)

        self._prev_obs = observation

    def export_memory(
        self,
        dot_path: Path,
        include_inactive: bool = False,
        png_path: Optional[Path] = None,
    ) -> None:
        """Optional visualization hook; overridden by agents that own graph memory."""
        return None

    def propose_action(
        self, obs: str, kg_text: str, recent_history: list[str], action_candidates: Optional[Sequence[str]] = None
    ) -> str:
        """
        Suggest the next action given the observation and memory context.
        Subclasses should override; default raises to surface unexpected usage.
        """
        raise NotImplementedError("propose_action is not implemented for BaseAgent.")

    def extract_entities_and_relations(
        self, prev_obs: str | None, action: str | None, obs: str
    ) -> list[CandidateFact]:
        """
        Default entity/relation extraction uses the heuristic extractor over the
        current world KG; subclasses can override to plug in LLM-based extractors.
        """
        if not self.use_memory or not self.entity_extractor or not self.world_kg:
            return []
        step = self._last_step if self._last_step is not None else 0
        return self.entity_extractor.extract(prev_obs, action, obs, self.world_kg, step)

    @abstractmethod
    def act(self, observation: str, action_candidates: list[str]) -> Optional[str]:
        """Return the next action string for the environment."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _init_memory_backend(self) -> None:
        """Create fresh memory components for a new episode."""
        self.world_kg = WorldKG()
        self.memory_manager = MemoryManager(self.world_kg)
        self.entity_extractor = NaiveEntityRelationExtractor()
        self.kg_snapshots = KGSnapshots()
        self.retriever = KGRetriever(self.world_kg)
        self._last_obs_node_id = None

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

    def _record_observation_transition(self, action: str, observation: str, step: int) -> None:
        """
        Add observation nodes and an ACTION edge linking previous -> current observation.
        """
        if not self.world_kg:
            return None
        if not action:
            action = "start game"
        curr_node_id = f"obs_{step}"
        self.world_kg.add_or_get_entity(
            canonical_id=curr_node_id,
            node_type="OBSERVATION",
            name=curr_node_id,
        )
        self.world_kg.graph.nodes[curr_node_id]["description"] = observation
        self.world_kg.graph.nodes[curr_node_id]["state"] = {"text": observation}
        self.world_kg.graph.nodes[curr_node_id]["last_updated_step"] = step

        if self._last_obs_node_id:
            try:
                self.world_kg.add_relation(
                    self._last_obs_node_id,
                    curr_node_id,
                    "ACTION",
                    command=action,
                    step=step,
                )
            except Exception:
                ...
        self._last_obs_node_id = curr_node_id
        return curr_node_id
