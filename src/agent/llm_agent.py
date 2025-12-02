from __future__ import annotations

"""Agent that queries an LLM to decide the next action and manages its own memory."""
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from llm import LLM
from memory import (
    ExtractedEntity,
    ExtractedEvent,
    ExtractedRelation,
    ExtractionResult,
    GraphStore,
    Grounder,
    build_extraction_prompt,
    parse_extraction_output,
)
from tools import export_graphviz

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
        super().__init__(memory=None)
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
        self._recent: List[Dict[str, str]] = []
        self._history: List[str] = []
        self.memory_store: Optional[GraphStore] = None
        self.grounder: Optional[Grounder] = None
        if self.memory_mode != "none":
            self._init_memory()

    def reset(self, env: object) -> None:
        super().reset(env)
        self._recent.clear()
        self._history.clear()
        if self.memory_mode != "none":
            self._init_memory()
        else:
            self.memory_store = None
            self.grounder = None

    def act(self, observation: str, action_candidates: List[str]) -> Optional[str]:
        prompt = self._build_prompt(observation, action_candidates)
        completion = self.llm.generate(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["\n"],
        )
        action = self._choose_from_candidates(completion, action_candidates)
        self._push_history(action, observation)
        return action or "look"

    def observe(
        self,
        turn_id: int,
        action: str,
        observation: str,
        reward: float,
        info: Optional[dict] = None,
    ) -> None:
        """Update in-memory graph from the completed step."""
        if action:
            self._history.append(f"{action} -> {observation}")

        if self.grounder and self.memory_store:
            if self.extraction_mode == "naive":
                extraction = self._naive_extract(observation, self._history, turn_id)
            else:
                prompt = build_extraction_prompt(observation, self._history[-5:])
                completion = self.llm.generate(
                    prompt,
                    max_tokens=self.extraction_max_tokens,
                    temperature=self.extraction_temperature,
                    stop=None,
                )
                extraction = parse_extraction_output(completion)
            self.grounder.ground(extraction, turn_id=turn_id, action=action)

    def export_memory(
        self,
        dot_path: Path,
        include_inactive: bool = False,
        png_path: Optional[Path] = None,
    ) -> None:
        """Visualize the current graph memory if enabled."""
        if not self.memory_store:
            return None
        return export_graphviz(
            self.memory_store,
            dot_path,
            include_inactive=include_inactive,
            render_png=bool(png_path),
            png_path=png_path,
        )

    def _build_prompt(self, observation: str, action_candidates: Sequence[str]) -> str:
        recent_lines = []
        for row in self._recent[-self.history_horizon :]:
            recent_lines.append(f"Action: {row['action']}")
            recent_lines.append(f"Observation: {row['observation']}")

        joined_history = "\n".join(recent_lines) if recent_lines else "None"
        options = "\n".join(f"- {a}" for a in action_candidates)
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

    def _init_memory(self) -> None:
        self.memory_store = GraphStore()
        self.grounder = Grounder(self.memory_store)
        self.memory = self.memory_store

    def _naive_extract(
        self, observation: str, history: List[str], turn_id: int
    ) -> ExtractionResult:
        """Lightweight heuristic extractor mirroring the smoke test."""
        entities: List[ExtractedEntity] = []
        relations: List[ExtractedRelation] = []
        events: List[ExtractedEvent] = []

        seen_entities = set()
        lowered = observation.lower()
        location_name: Optional[str] = None

        def add_entity(name: str, ent_type: str, confidence: float) -> None:
            if not name or name in seen_entities:
                return
            entities.append(
                ExtractedEntity(
                    name=name,
                    type=ent_type,
                    aliases=[name],
                    confidence=confidence,
                    time=turn_id,
                )
            )
            seen_entities.add(name)

        for marker in ["you are in ", "you are at "]:
            if marker in lowered:
                idx = lowered.index(marker) + len(marker)
                fragment = observation[idx:].split(".")[0].strip()
                if fragment:
                    add_entity(fragment, "location", 0.55)
                    location_name = fragment
                break

        for token in ["a ", "an ", "the "]:
            start = 0
            while True:
                idx = lowered.find(token, start)
                if idx == -1:
                    break
                chunk = (
                    observation[idx + len(token) :].split(",")[0].split(".")[0].strip()
                )
                chunk = " ".join(chunk.split()[:3])
                add_entity(chunk, "object", 0.45)
                start = idx + len(token)

        if location_name:
            for ent in entities:
                if ent.name == location_name:
                    continue
                relations.append(
                    ExtractedRelation(
                        source=location_name,
                        target=ent.name,
                        rel_label="contains",
                        confidence=0.4,
                        time=turn_id,
                    )
                )

        events.append(
            ExtractedEvent(
                description=observation[:180],
                participants=[e.name for e in entities],
                properties={},
                confidence=0.4,
                time=turn_id,
            )
        )

        return ExtractionResult(entities=entities, relations=relations, events=events)
