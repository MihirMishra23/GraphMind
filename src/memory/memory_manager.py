from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .canonicalizer import Canonicalizer
from .entity_extractor import CandidateFact
from .schema import EdgeType, NodeType, WorldKG


@dataclass
class MemoryDecision:
    action_type: str  # One of: WRITE_NEW, OVERWRITE_STATE, MOVE_ENTITY, NOOP
    reason: str


class MemoryManager:
    """
    Rule-based KG updater with a clear action space, ready to be swapped for an
    RL policy. Actions:
      - WRITE_NEW: add nodes/edges.
      - OVERWRITE_STATE: update state attributes.
      - MOVE_ENTITY: relocate entities (IN edges) and resolve conflicts.
      - NOOP: ignore the fact.
    """

    def __init__(self, world_kg: WorldKG):
        self.world_kg = world_kg
        self.canonicalizer = Canonicalizer(world_kg)

    def decide_and_apply(
        self, candidate_facts: list[CandidateFact], step: int, observation_node_id: Optional[str] = None
    ) -> list[MemoryDecision]:
        decisions: list[MemoryDecision] = []
        for fact in candidate_facts:
            decision = self._apply_fact(fact, step, observation_node_id)
            decisions.append(decision)
        return decisions

    def _apply_fact(self, fact: CandidateFact, step: int, observation_node_id: Optional[str]) -> MemoryDecision:
        # Simple confidence gate.
        if fact.confidence < 0.25:
            return MemoryDecision(action_type="NOOP", reason="Low confidence")

        rel = fact.rel_type.upper()
        if rel not in {et.value for et in EdgeType}:
            return MemoryDecision(action_type="NOOP", reason=f"Unknown relation {rel}")

        # Canonicalize subject / object ids with lightweight type hints.
        subj_type_hint = self._infer_type_hint(fact.subj_text, rel, role="subj")
        subj_ref = self.canonicalizer.canonicalize_entity_text(
            fact.subj_text, subj_type_hint.value, step=step
        )
        obj_ref: Optional[str] = None
        obj_type_hint: Optional[NodeType] = None
        if fact.obj_text:
            obj_type_hint = self._infer_type_hint(fact.obj_text, rel, role="obj")
            obj_ref = self.canonicalizer.canonicalize_entity_text(
                fact.obj_text, obj_type_hint.value, step=step
            ).canonical_id

        try:
            if observation_node_id:
                self._link_observation(observation_node_id, subj_ref.canonical_id, obj_ref)

            if rel == EdgeType.STATE.value:
                if fact.state_updates:
                    for key, value in fact.state_updates.items():
                        self.world_kg.set_state(subj_ref.canonical_id, key, value, step)
                    return MemoryDecision(
                        action_type="OVERWRITE_STATE", reason="Applied state updates"
                    )
                if obj_ref:
                    self.world_kg.add_relation(subj_ref.canonical_id, obj_ref, rel)
                    return MemoryDecision(action_type="WRITE_NEW", reason="Added STATE edge")
                return MemoryDecision(action_type="NOOP", reason="No state updates or flag target")

            if rel in {EdgeType.IN.value, EdgeType.HAS.value, EdgeType.CONTAINS.value} and obj_ref:
                # Treat CONTAINS as generalized containment.
                if subj_ref.node_type == NodeType.PLAYER.value:
                    self._set_has(subj_ref.canonical_id, obj_ref, step)
                    return MemoryDecision(action_type="WRITE_NEW", reason="Set inventory ownership")
                if subj_ref.node_type in {NodeType.OBJECT.value, NodeType.CHARACTER.value}:
                    self._remove_has_edges(subj_ref.canonical_id)
                    self.world_kg.move_entity(subj_ref.canonical_id, obj_ref, step)
                    return MemoryDecision(action_type="MOVE_ENTITY", reason="Moved entity into container")
                try:
                    self.world_kg.add_relation(subj_ref.canonical_id, obj_ref, EdgeType.CONTAINS.value)
                    return MemoryDecision(action_type="WRITE_NEW", reason="Linked observation/entity with CONTAINS")
                except Exception as exc:
                    return MemoryDecision(action_type="NOOP", reason=f"Failed CONTAINS: {exc}")

            if rel == EdgeType.ON.value and obj_ref:
                self.world_kg.add_relation(subj_ref.canonical_id, obj_ref, rel)
                return MemoryDecision(action_type="WRITE_NEW", reason="Placed object on surface")

            if rel == EdgeType.MENTIONS.value and obj_ref:
                self.world_kg.add_relation(subj_ref.canonical_id, obj_ref, rel)
                return MemoryDecision(action_type="WRITE_NEW", reason="Recorded note mention")

            if rel == EdgeType.CONNECTED_TO.value and obj_ref:
                direction = fact.state_updates.get("direction") if fact.state_updates else None
                attrs = {}
                if direction:
                    attrs["direction"] = direction
                else:
                    # Default placeholder direction for schema compliance.
                    attrs["direction"] = "north"
                self.world_kg.add_relation(subj_ref.canonical_id, obj_ref, rel, **attrs)
                return MemoryDecision(action_type="WRITE_NEW", reason="Linked rooms")

        except Exception as exc:  # pragma: no cover - defensive
            return MemoryDecision(action_type="NOOP", reason=f"Failed to apply fact: {exc}")

        return MemoryDecision(action_type="NOOP", reason="Unhandled fact shape")

    # ------------------------------------------------------------------ #
    # Helper routines
    # ------------------------------------------------------------------ #
    def _infer_type_hint(self, text: str, rel: str, role: str) -> NodeType:
        lowered = text.lower().strip()
        if lowered in {"player", "you", "yourself"}:
            return NodeType.PLAYER
        if "note" in lowered or "clue" in lowered or rel == EdgeType.MENTIONS.value:
            return NodeType.NOTE
        if rel == EdgeType.CONNECTED_TO.value:
            return NodeType.ROOM
        if role == "obj" and rel in {EdgeType.IN.value, EdgeType.CONTAINS.value}:
            if any(k in lowered for k in ["room", "hall", "chamber", "kitchen"]):
                return NodeType.ROOM
            return NodeType.OBJECT
        if rel == EdgeType.HAS.value and role == "subj" and lowered not in {"player", "you", "yourself"}:
            return NodeType.CHARACTER
        return NodeType.OBJECT

    def _link_observation(self, obs_node_id: str, subj_id: str, obj_id: Optional[str]) -> None:
        """
        Ensure the creating/editing observation is linked to involved entities via CONTAINS.
        """
        if obs_node_id not in self.world_kg.graph:
            return
        obs_type = self.world_kg.graph.nodes[obs_node_id].get("type")
        if obs_type != NodeType.OBSERVATION.value:
            return

        for eid in [subj_id, obj_id]:
            if not eid or eid not in self.world_kg.graph:
                continue
            try:
                self.world_kg.add_relation(obs_node_id, eid, EdgeType.CONTAINS.value)
            except Exception:
                continue

    def _remove_has_edges(self, entity_id: str) -> None:
        # Remove incoming HAS edges (prevent object being both in room and owned).
        for src, _, key, data in list(self.world_kg.graph.in_edges(entity_id, keys=True, data=True)):
            if data.get("type") == EdgeType.HAS.value:
                self.world_kg.graph.remove_edge(src, entity_id, key=key)

    def _set_has(self, owner_id: str, obj_id: str, step: int) -> None:
        # Remove other owners and IN edges so the object is only in inventory.
        self._remove_has_edges(obj_id)
        for _, dst, key, data in list(self.world_kg.graph.out_edges(obj_id, keys=True, data=True)):
            if data.get("type") == EdgeType.IN.value:
                self.world_kg.graph.remove_edge(obj_id, dst, key=key)
        self.world_kg.add_relation(owner_id, obj_id, EdgeType.HAS.value)
        self.world_kg.graph.nodes[obj_id]["last_updated_step"] = step

    def get_rl_state_representation(self, fact: CandidateFact) -> dict:
        """
        Extract a compact, model-agnostic representation of the decision
        context. Intended to be fed to a future policy.
        """
        subj_exists = fact.subj_text in self.world_kg.graph
        obj_exists = fact.obj_text in self.world_kg.graph if fact.obj_text else False
        subj_type = (
            self.world_kg.graph.nodes[fact.subj_text]["type"] if subj_exists else self._infer_type_hint(fact.subj_text, fact.rel_type, "subj").value
        )
        obj_type = (
            self.world_kg.graph.nodes[fact.obj_text]["type"] if fact.obj_text and obj_exists else (self._infer_type_hint(fact.obj_text, fact.rel_type, "obj").value if fact.obj_text else None)
        )
        return {
            "rel_type": fact.rel_type,
            "source": fact.source,
            "confidence": fact.confidence,
            "subj_type": subj_type,
            "obj_type": obj_type,
            "subj_exists": subj_exists,
            "obj_exists": obj_exists,
            "has_state_updates": bool(fact.state_updates),
        }
