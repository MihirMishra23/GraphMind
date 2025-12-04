from __future__ import annotations

import json
import re
import textwrap
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from llm import LLM

from .schema import WorldKG


@dataclass
class CandidateFact:
    subj_text: str
    rel_type: str
    obj_text: Optional[str]
    state_updates: Dict[str, Any]  # e.g. {"open": True}
    source: str  # "obs" or "action"
    confidence: float


logger = logging.getLogger(__name__)


class EntityRelationExtractor:
    """
    Abstract extractor interface to produce candidate facts.
    """

    def extract(
        self,
        prev_obs: str | None,
        action: str | None,
        obs: str,
        world_kg: WorldKG,
        step: int,
    ) -> list[CandidateFact]:
        raise NotImplementedError


class NaiveEntityRelationExtractor(EntityRelationExtractor):
    """
    Lightweight, deterministic extractor for text-based game observations.
    """

    def __init__(self):
        self.room_patterns = [
            re.compile(r"\byou are in (?:the )?(?P<room>[\w\s'-]+)", re.IGNORECASE),
            re.compile(r"\bthis is (?:the )?(?P<room>[\w\s'-]+)", re.IGNORECASE),
        ]
        self.see_patterns = [
            re.compile(r"\byou see (.*)", re.IGNORECASE),
            re.compile(r"\bthere (?:is|are) (.*)", re.IGNORECASE),
            re.compile(r"\bhere (?:is|are) (.*)", re.IGNORECASE),
        ]
        self.state_pattern = re.compile(
            r"(?P<target>[\w\s'-]+?)\s+(?:is|are)\s+(?P<state>open|closed|locked|unlocked|lit|dark)\b",
            re.IGNORECASE,
        )
        self.on_pattern = re.compile(
            r"on the (?P<surface>[\w\s'-]+) is (?:an?|the )?(?P<item>[\w\s'-]+)",
            re.IGNORECASE,
        )

    def extract(
        self,
        prev_obs: str | None,
        action: str | None,
        obs: str,
        world_kg: WorldKG,
        step: int,
    ) -> list[CandidateFact]:
        """
        Use simple pattern matching + heuristics to extract:

        - Current room
        - Objects in the room
        - Inventory changes
        - Door/open/closed/locked states
        - Notes/Clues

        For now, implement a deterministic heuristic system; leave clear extension
        points for future LLM-based extraction.
        """
        facts: list[CandidateFact] = []
        obs_lines = [line.strip() for line in obs.splitlines() if line.strip()]

        room_name = self._extract_room_name(obs)
        if room_name:
            facts.append(
                CandidateFact(
                    subj_text="player",
                    rel_type="IN",
                    obj_text=room_name,
                    state_updates={},
                    source="obs",
                    confidence=0.9,
                )
            )

        facts.extend(self._extract_inventory(obs, room_name))
        facts.extend(self._extract_seen_objects(obs_lines, room_name))
        facts.extend(self._extract_on_relations(obs_lines))
        facts.extend(self._extract_states(obs_lines))
        facts.extend(self._extract_from_action(action, room_name))
        facts.extend(self._extract_notes(obs_lines))
        facts.extend(self._extract_entity_mentions(obs, world_kg))

        return facts

    def _extract_room_name(self, text: str) -> Optional[str]:
        for pattern in self.room_patterns:
            match = pattern.search(text)
            if match:
                room = match.group("room").strip(" .")
                return room
        return None

    def _split_items(self, text: str) -> list[str]:
        cleaned = text.rstrip(".")
        parts = re.split(r",| and ", cleaned)
        return [p.strip() for p in parts if p.strip()]

    def _extract_inventory(
        self, obs: str, room_name: Optional[str]
    ) -> list[CandidateFact]:
        facts: list[CandidateFact] = []
        lines = obs.splitlines()
        inventory_start = None
        for idx, line in enumerate(lines):
            if re.search(r"\byou (?:are )?carrying\b", line, re.IGNORECASE):
                inventory_start = idx
                break
        if inventory_start is None:
            return facts

        items: list[str] = []
        first_line = lines[inventory_start]
        after_colon = first_line.split(":", 1)
        if len(after_colon) > 1 and after_colon[1].strip():
            items.extend(self._split_items(after_colon[1]))

        for line in lines[inventory_start + 1 :]:
            stripped = line.strip()
            if not stripped:
                break
            if stripped.startswith(("-", "*")):
                stripped = stripped[1:].strip()
                items.extend(self._split_items(stripped))
            else:
                break

        for item in items:
            facts.append(
                CandidateFact(
                    subj_text="player",
                    rel_type="HAS",
                    obj_text=item,
                    state_updates={},
                    source="obs",
                    confidence=0.85,
                )
            )
        return facts

    def _extract_seen_objects(
        self, obs_lines: list[str], room_name: Optional[str]
    ) -> list[CandidateFact]:
        facts: list[CandidateFact] = []
        for line in obs_lines:
            for pattern in self.see_patterns:
                match = pattern.search(line)
                if not match:
                    continue
                items_text = match.group(1)
                items = self._split_items(items_text)
                for item in items:
                    item_clean = re.sub(
                        r"\bhere\b", "", item, flags=re.IGNORECASE
                    ).strip()
                    facts.append(
                        CandidateFact(
                            subj_text=item_clean,
                            rel_type="IN",
                            obj_text=room_name,
                            state_updates={},
                            source="obs",
                            confidence=0.65,
                        )
                    )
        return facts

    def _extract_on_relations(self, obs_lines: list[str]) -> list[CandidateFact]:
        facts: list[CandidateFact] = []
        for line in obs_lines:
            match = self.on_pattern.search(line)
            if not match:
                continue
            surface = match.group("surface").strip()
            item = match.group("item").strip()
            facts.append(
                CandidateFact(
                    subj_text=item,
                    rel_type="ON",
                    obj_text=surface,
                    state_updates={},
                    source="obs",
                    confidence=0.7,
                )
            )
        return facts

    def _extract_states(self, obs_lines: list[str]) -> list[CandidateFact]:
        facts: list[CandidateFact] = []
        for line in obs_lines:
            match = self.state_pattern.search(line)
            if not match:
                continue
            target = match.group("target").strip()
            state_word = match.group("state").lower()
            updates: Dict[str, Any] = {}
            if state_word in {"open", "closed"}:
                updates["open"] = state_word == "open"
            if state_word in {"locked", "unlocked"}:
                updates["locked"] = state_word == "locked"
            if state_word in {"lit", "dark"}:
                updates["lit"] = state_word == "lit"
            facts.append(
                CandidateFact(
                    subj_text=target,
                    rel_type="STATE",
                    obj_text=None,
                    state_updates=updates,
                    source="obs",
                    confidence=0.6,
                )
            )
        return facts

    def _extract_from_action(
        self, action: Optional[str], room_name: Optional[str]
    ) -> list[CandidateFact]:
        if not action:
            return []
        facts: list[CandidateFact] = []
        action_lower = action.lower()
        take_match = re.search(
            r"(take|pick up|grab)\s+(?P<item>[\w\s'-]+)", action_lower
        )
        if take_match:
            item = take_match.group("item").strip()
            facts.append(
                CandidateFact(
                    subj_text="player",
                    rel_type="HAS",
                    obj_text=item,
                    state_updates={},
                    source="action",
                    confidence=0.7,
                )
            )

        drop_match = re.search(r"(drop|leave)\s+(?P<item>[\w\s'-]+)", action_lower)
        if drop_match:
            item = drop_match.group("item").strip()
            facts.append(
                CandidateFact(
                    subj_text=item,
                    rel_type="IN",
                    obj_text=room_name,
                    state_updates={},
                    source="action",
                    confidence=0.55,
                )
            )

        open_match = re.search(r"(open|unlock)\s+(?P<item>[\w\s'-]+)", action_lower)
        if open_match:
            item = open_match.group("item").strip()
            facts.append(
                CandidateFact(
                    subj_text=item,
                    rel_type="STATE",
                    obj_text=None,
                    state_updates={"open": True},
                    source="action",
                    confidence=0.55,
                )
            )
        close_match = re.search(r"(close|lock)\s+(?P<item>[\w\s'-]+)", action_lower)
        if close_match:
            item = close_match.group("item").strip()
            state_updates: Dict[str, Any] = {"open": False}
            if close_match.group(1) == "lock":
                state_updates["locked"] = True
            facts.append(
                CandidateFact(
                    subj_text=item,
                    rel_type="STATE",
                    obj_text=None,
                    state_updates=state_updates,
                    source="action",
                    confidence=0.55,
                )
            )
        return facts

    def _extract_notes(self, obs_lines: list[str]) -> list[CandidateFact]:
        facts: list[CandidateFact] = []
        for line in obs_lines:
            match = re.search(
                r"\b(note|clue|message):?\s+(?P<content>.+)", line, re.IGNORECASE
            )
            if not match:
                continue
            content = match.group("content").strip()
            facts.append(
                CandidateFact(
                    subj_text="note",
                    rel_type="STATE",
                    obj_text=None,
                    state_updates={"text": content},
                    source="obs",
                    confidence=0.4,
                )
            )
        return facts

    def _extract_entity_mentions(
        self, obs: str, world_kg: WorldKG
    ) -> list[CandidateFact]:
        """
        Mark mentions of known entities to help retrieval. Uses simple substring
        matches against existing node names.
        """
        obs_lower = obs.lower()
        facts: list[CandidateFact] = []
        for node_id, data in world_kg.graph.nodes(data=True):
            name = data.get("name", node_id)
            if not name:
                continue
            if name.lower() in obs_lower:
                facts.append(
                    CandidateFact(
                        subj_text="note",
                        rel_type="MENTIONS",
                        obj_text=name,
                        state_updates={},
                        source="obs",
                        confidence=0.5,
                    )
                )
        return facts


class LLMEntityRelationExtractor(EntityRelationExtractor):
    """
    LLM-based extractor.
    """

    def __init__(
        self,
        llm: LLM,
        max_tokens: int = 256,
    ):
        self.llm = llm
        self.max_tokens = max_tokens

    def extract(
        self,
        prev_obs: str | None,
        action: str | None,
        obs: str,
        world_kg: WorldKG,
        step: int,
    ) -> list[CandidateFact]:
        prompt = self._build_prompt(prev_obs, action, obs)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("LLM extractor prompt (step %s):\n%s", step, prompt)
        completion = self.llm.generate(
            prompt,
            max_tokens=self.max_tokens,
            stop=["\n\n", "\nReturn", "\nNext", "\nNote"],
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "LLM extractor completion (step %s, truncated): %s",
                step,
                completion[:500],
            )
        facts = self._parse_completion(completion)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "LLM extractor parsed %d candidate facts at step %s", len(facts), step
            )
        return facts

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _build_prompt(self, prev_obs: str | None, action: str | None, obs: str) -> str:
        prev_block = prev_obs if prev_obs else "None"
        action_block = action if action else "None"
        template = textwrap.dedent(
            """
            You read observations and propose structured facts about the world.
            Return ONLY a JSON array of objects with keys:
              subj_text (string), rel_type (CONTAINS|ON|STATE|CONNECTED_TO),
              obj_text (string or null), state_updates (object), source ("obs"|"action"),
              confidence (0-1).

            Example:
            [
              {{"subj_text": "mailbox", "rel_type": "CONTAINS", "obj_text": "leaflet", "state_updates": {{}}, "source": "obs", "confidence": 0.8}}
            ]

            Previous observation: {prev_obs}
            Previous action: {action}
            Current observation: {obs}

            JSON array:
            """
        ).strip()
        return template.format(prev_obs=prev_block, action=action_block, obs=obs)

    def _parse_completion(self, completion: str) -> list[CandidateFact]:
        """Parse a JSON array from the LLM; fall back to empty on errors."""
        raw = completion.strip()
        payload = self._extract_json_array(raw)
        if payload is None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Failed to parse LLM extractor completion as JSON")
            return []
        try:
            data = json.loads(payload)
        except Exception:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Failed to parse extracted JSON array")
            return []

        facts: list[CandidateFact] = []
        if not isinstance(data, list):
            return facts

        for entry in data:
            if not isinstance(entry, dict):
                continue
            subj = entry.get("subj_text")
            rel = entry.get("rel_type")
            obj = entry.get("obj_text")
            state_updates = entry.get("state_updates") or {}
            source = entry.get("source") or "obs"
            confidence = float(entry.get("confidence", 0.5))
            if not subj or not rel:
                continue
            facts.append(
                CandidateFact(
                    subj_text=str(subj),
                    rel_type=str(rel),
                    obj_text=str(obj) if obj is not None else None,
                    state_updates=(
                        state_updates if isinstance(state_updates, dict) else {}
                    ),
                    source=str(source),
                    confidence=max(0.0, min(1.0, confidence)),
                )
            )
        return facts

    def _extract_json_array(self, text: str) -> Optional[str]:
        """
        Try to extract the first JSON array from the text, handling fenced code blocks.
        """
        import re

        # Prefer fenced ```json ... ``` blocks.
        fenced = re.findall(r"```json\\s*(\\[.*?\\])\\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        for block in fenced:
            if block.strip():
                return block.strip()

        # Fallback: grab from first '[' to last ']'.
        if "[" in text and "]" in text:
            start = text.find("[")
            end = text.rfind("]")
            if end > start:
                return text[start : end + 1].strip()
        return None
