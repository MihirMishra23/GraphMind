from __future__ import annotations

"""LLM agent that consumes a structured KG context to choose actions."""
from typing import List, Optional, Sequence
import re

from llm import LLM
from memory.schema import EdgeType, NodeType, WorldKG

from .llm_agent import LLMAgent, DEFAULT_SYSTEM_PROMPT
from .state_effect_model import StateEffectModel


STRUCTURED_SYSTEM_PROMPT = """You control a player in a text adventure game. Your goal is to explore, solve puzzles, collect treasures, and win.

Each turn you receive:
- Recent history (last ~10 turns) of actions and observations.
- Structured world state from the knowledge graph (location, exits, objects, inventory, flags).
- The latest observation resulting from your last action.

Your task:
- Think step by step about the best next action using the structured context, latest observation, last action, and recent history.
- If unsure, prefer exploring new states (new directions, inspecting new objects, “look”/“inventory”) rather than repeating loops.
- Avoid random or purposeless moves; favor progress toward exploration and puzzles.
- After reasoning, output exactly ONE concise game command (1–3 words).

Output format:
- Include your reasoning.
- End with the final command wrapped exactly as:
  <start> your command <end>"""


class KGLLMAgent(LLMAgent):
    """
    Extension of LLMAgent that feeds a structured KG context (location, exits,
    inventory, objects) into the prompt for more state-aware decisions.
    """

    def __init__(
        self,
        llm: LLM,
        memory_mode: str = "llama",
        system_prompt: str = STRUCTURED_SYSTEM_PROMPT,
        **kwargs,
    ) -> None:
        super().__init__(
            llm=llm, memory_mode=memory_mode, system_prompt=system_prompt, **kwargs
        )
        self.state_effect_model = StateEffectModel(llm)

    def propose_action(
        self,
        obs: str,
        kg_text: str,
        recent_history: list[str],
        action_candidates: Optional[Sequence[str]] = None,
    ) -> str:
        history_block = (
            "\n".join(recent_history[-self.history_horizon :])
            if recent_history
            else "None"
        )
        kg_block = self._build_structured_context(obs)
        prompt = (
            f"{self.system_prompt}\n\n"
            f"Recent history:\n{history_block}\n\n"
            f"World state:\n{kg_block}\n\n"
            f"Latest observation:\n{obs}\n\n"
        )
        if action_candidates:
            options = "\n".join(f"- {a}" for a in action_candidates)
            prompt += f"Allowed actions (choose one):\n{options}\n\n"
        prompt += "Next action (one command):"

        completion = self.llm.generate(
            prompt,
            stop=["\n"],
        )
        completion = completion.strip()
        if not action_candidates:
            return completion

        lines = completion.splitlines()
        cleaned = lines[0] if lines else ""
        cleaned_lower = cleaned.lower()
        for cand in action_candidates:
            if cleaned_lower == cand.lower():
                return cand
        for cand in action_candidates:
            if cleaned_lower.startswith(cand.lower()) or cand.lower().startswith(
                cleaned_lower
            ):
                return cand
        return action_candidates[0]

    # ------------------------------------------------------------------ #
    # KG context helpers
    # ------------------------------------------------------------------ #
    def _apply_naive_outcome_update(self, action: str, observation: str, step: int) -> None:
        """
        LLM-driven effect updates for terse result-only observations.
        TODO: Replace with dedicated effect model; keep separate from episodic KG prompts.
        """
        if not self.world_kg or not self.memory_manager:
            return
        action_lower = action.strip().lower()
        obs_clean = observation.strip().lower()
        if not self._is_result_only_observation(obs_clean):
            return

        # Build a lightweight facts list for the effect model.
        facts: list[dict] = [
            {
                "subj_text": f.subj_text,
                "rel_type": f.rel_type,
                "obj_text": f.obj_text,
                "state_updates": f.state_updates,
                "source": f.source,
                "confidence": f.confidence,
            }
            for f in self.extract_entities_and_relations(None, action, observation)
        ]
        effects = self.state_effect_model.propose_effects(action_lower, observation, facts)

        for eff in effects:
            if not isinstance(eff, dict):
                continue
            op = eff.get("op")
            target = eff.get("target")
            state_updates = eff.get("state_updates") or {}
            container = eff.get("container")
            confidence = float(eff.get("confidence", 0.0))
            if confidence < 0.4 or not target:
                continue

            if op == "add_inventory":
                player_ref = self.memory_manager.canonicalizer.canonicalize_entity_text(
                    "player", NodeType.PLAYER.value, step=step
                ).canonical_id
                obj_ref = self.memory_manager.canonicalizer.canonicalize_entity_text(
                    target, NodeType.OBJECT.value, step=step
                ).canonical_id
                self.memory_manager._set_has(player_ref, obj_ref, step)

            if op == "set_state":
                obj_ref = self.memory_manager.canonicalizer.canonicalize_entity_text(
                    target, NodeType.OBJECT.value, step=step
                ).canonical_id
                for key, value in (state_updates if isinstance(state_updates, dict) else {}).items():
                    self.world_kg.set_state(obj_ref, key, value, step)

            if op == "contains" and container:
                cont_ref = self.memory_manager.canonicalizer.canonicalize_entity_text(
                    container, NodeType.OBJECT.value, step=step
                ).canonical_id
                obj_ref = self.memory_manager.canonicalizer.canonicalize_entity_text(
                    target, NodeType.OBJECT.value, step=step
                ).canonical_id
                try:
                    self.world_kg.add_relation(cont_ref, obj_ref, EdgeType.CONTAINS.value)
                except Exception:
                    continue

    def _build_structured_context(self, observation: str) -> str:
        """
        Build a concise, structured view of the relevant KG neighborhood:
        location, exits, inventory, objects present, notes/flags.
        """
        if not self.use_memory or not self.retriever or not self.world_kg:
            return "No structured world state available."
        step = self._last_step if self._last_step is not None else 0
        subgraph: WorldKG = self.retriever.get_relevant_subgraph(
            observation, step=step, radius=2
        )

        player_ids = [
            n
            for n, d in subgraph.graph.nodes(data=True)
            if d.get("type") == NodeType.PLAYER.value
        ]
        player_id = player_ids[0] if player_ids else None
        location_id = None
        if player_id:
            location_id = (
                subgraph.graph.nodes[player_id].get("state", {}).get("location")
            )

        lines: list[str] = []
        # Location + contents
        if location_id and location_id in subgraph.graph:
            loc_data = subgraph.graph.nodes[location_id]
            lines.append(
                f"Location: {loc_data.get('name', location_id)} {self._format_state(loc_data)}"
            )
            contained = []
            for src, dst, data in subgraph.graph.in_edges(location_id, data=True):
                if data.get("type") == EdgeType.IN.value:
                    contained.append(
                        f"{subgraph.graph.nodes[src].get('name', src)} {self._format_state(subgraph.graph.nodes[src])}".strip()
                    )
            if contained:
                lines.append("Objects here: " + "; ".join(contained))
            exits = []
            for src, dst, data in subgraph.graph.edges(location_id, data=True):
                if data.get("type") == EdgeType.CONNECTED_TO.value:
                    exits.append(
                        f"{data.get('direction', '?')} -> {subgraph.graph.nodes[dst].get('name', dst)}"
                    )
            if exits:
                lines.append("Exits: " + "; ".join(exits))
        else:
            lines.append("Location: Unknown")

        # Inventory
        if player_id:
            inventory = []
            for _, obj, data in subgraph.graph.edges(player_id, data=True):
                if data.get("type") == EdgeType.HAS.value and obj in subgraph.graph:
                    inventory.append(
                        f"{subgraph.graph.nodes[obj].get('name', obj)} {self._format_state(subgraph.graph.nodes[obj])}".strip()
                    )
            if inventory:
                lines.append("Inventory: " + "; ".join(inventory))

        # Notes / flags
        notes: list[str] = []
        for node_id, data in subgraph.graph.nodes(data=True):
            if data.get("type") in {NodeType.NOTE.value, NodeType.FLAG.value}:
                note_state = self._format_state(data)
                label = data.get("name", node_id)
                notes.append(f"{label} {note_state}".strip())
        if notes:
            lines.append("Notes/Flags: " + "; ".join(notes))

        return "\n".join(lines) if lines else "World state unavailable."

    def _format_state(self, data: dict) -> str:
        state = data.get("state") or {}
        if not state:
            return ""
        return "(" + "; ".join(f"{k}={v}" for k, v in state.items()) + ")"
