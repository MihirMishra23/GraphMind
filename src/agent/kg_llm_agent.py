from __future__ import annotations

"""LLM agent that consumes a structured KG context to choose actions."""
from typing import List, Optional, Sequence

from llm import LLM
from memory.schema import EdgeType, NodeType, WorldKG

from .llm_agent import LLMAgent, DEFAULT_SYSTEM_PROMPT


STRUCTURED_SYSTEM_PROMPT = """You control a player in a text adventure game.
Use the structured world state and recent history to pick the single best next action.
Return only the command text (no quotes or explanations)."""


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
