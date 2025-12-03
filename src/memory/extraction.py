"""Extraction and grounding utilities for Jericho observations."""

from __future__ import annotations

import json
import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .graph_store import GraphStore

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    name: str
    type: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    confidence: float = 1.0
    time: Optional[int] = None


@dataclass
class ExtractedRelation:
    source: str
    target: str
    rel_label: str
    evidence: Optional[str] = None
    confidence: float = 1.0
    time: Optional[int] = None


@dataclass
class ExtractedEvent:
    description: str
    participants: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    time: Optional[int] = None


@dataclass
class ExtractionResult:
    entities: List[ExtractedEntity] = field(default_factory=list)
    relations: List[ExtractedRelation] = field(default_factory=list)
    events: List[ExtractedEvent] = field(default_factory=list)


EXTRACTION_PROMPT_HEADER = """You are an information extractor for parser-based games.
Read the recent action/observation pair and emit concise JSON capturing entities, relations,
and events. Prefer open-vocabulary labels over schemas.

Schema:
- entities: name, type (location/entity/person/etc), aliases, properties, confidence [0-1], time (turn id)
- relations: source, target, rel_label, evidence (short span), confidence [0-1], time
- events: description, participants (entity names), properties (tense/status), confidence, time

Rules:
- Only include facts grounded in the text; no speculation.
- Keep names short and stable; reuse earlier entity strings when possible.
- Output pure JSON with keys: entities, relations, events."""

FEW_SHOT_EXAMPLES = """
Example:
History: ["look -> You are in the kitchen. A dusty table stands here.", "get lamp -> Taken."]
Observation: "A closed wooden door leads north. You feel hungry."
Output:
{
  "entities": [
    {"name": "kitchen", "type": "location", "aliases": ["kitchen"], "confidence": 0.74},
    {"name": "wooden door", "type": "entity", "aliases": ["door"], "confidence": 0.62}
  ],
  "relations": [
    {"source": "kitchen", "target": "wooden door", "rel_label": "connects_north", "confidence": 0.55}
  ],
  "events": []
}
"""


def build_extraction_prompt(
    observation: str, recent_history: Optional[List[str]] = None
) -> str:
    """Construct a few-shot prompt for the LLaMA IE model."""
    history_text = "\n".join(recent_history or [])
    return (
        f"{EXTRACTION_PROMPT_HEADER}\n\n"
        f"{FEW_SHOT_EXAMPLES}\n"
        f"History: [{history_text}]\n"
        f'Observation: "{observation}"\n'
        "Output JSON:"
    )


def parse_extraction_output(text: str) -> ExtractionResult:
    """Parse an LLM extraction completion into structured objects."""
    if not text:
        return ExtractionResult()
    payload = _extract_json_block(text)
    try:
        data = json.loads(payload)
    except Exception:
        return ExtractionResult()

    result = ExtractionResult()
    result.entities = _parse_entities(data.get("entities"))
    result.relations = _parse_relations(data.get("relations"))
    result.events = _parse_events(data.get("events"))
    return result


def _extract_json_block(text: str) -> str:
    fence_match = re.search(
        r"```(?:json)?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE
    )
    if fence_match:
        return fence_match.group(1).strip()
    brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace_match:
        return brace_match.group(0)
    return text.strip()


def _parse_entities(raw: Any) -> List[ExtractedEntity]:
    entities: List[ExtractedEntity] = []
    if not isinstance(raw, list):
        return entities
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = _clean_str(item.get("name") or item.get("entity"))
        if not name:
            continue
        aliases = [a for a in item.get("aliases", []) if isinstance(a, str)]
        embedding = _parse_embedding(item.get("embedding"))
        ent = ExtractedEntity(
            name=name,
            type=_clean_str(item.get("type") or item.get("category")),
            aliases=aliases,
            properties=_normalize_properties(item.get("properties")),
            embedding=embedding,
            confidence=_to_float(item.get("confidence"), 1.0),
            time=_to_int(item.get("time")),
        )
        entities.append(ent)
    return entities


def _parse_relations(raw: Any) -> List[ExtractedRelation]:
    relations: List[ExtractedRelation] = []
    if not isinstance(raw, list):
        return relations
    for item in raw:
        if not isinstance(item, dict):
            continue
        source = _clean_str(item.get("source"))
        target = _clean_str(item.get("target"))
        rel_label = _clean_str(item.get("rel_label") or item.get("relation"))
        if not (source and target and rel_label):
            continue
        relations.append(
            ExtractedRelation(
                source=source,
                target=target,
                rel_label=rel_label,
                evidence=_clean_str(item.get("evidence")),
                confidence=_to_float(item.get("confidence"), 1.0),
                time=_to_int(item.get("time")),
            )
        )
    return relations


def _parse_events(raw: Any) -> List[ExtractedEvent]:
    events: List[ExtractedEvent] = []
    if not isinstance(raw, list):
        return events
    for item in raw:
        if not isinstance(item, dict):
            continue
        desc = _clean_str(item.get("description") or item.get("event"))
        if not desc:
            continue
        participants = [p for p in item.get("participants", []) if isinstance(p, str)]
        events.append(
            ExtractedEvent(
                description=desc,
                participants=participants,
                properties=_normalize_properties(item.get("properties")),
                confidence=_to_float(item.get("confidence"), 1.0),
                time=_to_int(item.get("time")),
            )
        )
    return events


def _parse_embedding(raw: Any) -> Optional[List[float]]:
    if not isinstance(raw, list):
        return None
    cleaned: List[float] = []
    for val in raw:
        try:
            cleaned.append(float(val))
        except (TypeError, ValueError):
            continue
    return cleaned or None


def _to_float(val: Any, default: float) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _to_int(val: Any) -> Optional[int]:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _clean_str(val: Any) -> str:
    return str(val).strip() if isinstance(val, str) else ""


def _normalize_properties(raw: Any) -> Dict[str, Any]:
    """Coerce loose LLM outputs into a property dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        return {"items": raw}
    if isinstance(raw, str):
        return {"value": raw}
    return {}


@dataclass
class GroundedUpdate:
    entity_nodes: Dict[str, str] = field(default_factory=dict)
    event_nodes: Dict[str, str] = field(default_factory=dict)
    relation_edges: List[str] = field(default_factory=list)


class Grounder:
    """Ground extracted tuples into the graph store with aliasing."""

    def __init__(self, store: GraphStore, embedding_threshold: float = 0.8) -> None:
        self.store = store
        self.embedding_threshold = embedding_threshold
        self.last_event_node: Optional[str] = None

    def ground(
        self,
        extraction: ExtractionResult,
        turn_id: Optional[int] = None,
        action: Optional[str] = None,
    ) -> GroundedUpdate:
        entity_map = self._ground_entities(extraction.entities, turn_id)
        event_map = self._ground_events(extraction.events, entity_map, turn_id)
        relation_edges = self._ground_relations(
            extraction.relations, entity_map, turn_id
        )

        if action and event_map:
            current_event_id = next(iter(event_map.values()))
            if self.last_event_node:
                existing = next(
                    (
                        e
                        for e in self.store.active_edges()
                        if e.source == self.last_event_node
                        and e.target == current_event_id
                        and e.rel_label == action
                    ),
                    None,
                )
                if existing:
                    logger.debug(
                        "Reusing action edge %s (%s -[%s]-> %s)",
                        existing.edge_id,
                        self.last_event_node,
                        action,
                        current_event_id,
                    )
                else:
                    edge = self.store.add_edge(
                        source=self.last_event_node,
                        target=current_event_id,
                        rel_label=action,
                        provenance={"source": "action"},
                        valid_from=turn_id,
                    )
                    logger.debug(
                        "Created action edge %s (%s -[%s]-> %s) at turn=%s",
                        edge.edge_id,
                        self.last_event_node,
                        action,
                        current_event_id,
                        turn_id,
                    )
            self.last_event_node = current_event_id

        return GroundedUpdate(
            entity_nodes=entity_map,
            event_nodes=event_map,
            relation_edges=relation_edges,
        )

    def _ground_entities(
        self, entities: List[ExtractedEntity], turn_id: Optional[int]
    ) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for ent in entities:
            aliases = [ent.name] + [a for a in ent.aliases if a]
            candidates: List[str] = []
            for alias in aliases:
                candidates.extend(self.store.lookup_surface(alias))
            roots = (
                {self.store.find_root(cid) for cid in candidates}
                if candidates
                else set()
            )

            chosen: Optional[str] = None
            if roots:
                chosen = next(iter(roots))
                for other in list(roots)[1:]:
                    chosen = self.store.merge_aliases(chosen, other)
                logger.debug(
                    "Reusing entity node %s for aliases=%s (roots=%s, merged=%s)",
                    chosen,
                    aliases,
                    roots,
                    chosen,
                )

            if chosen is None and ent.embedding:
                hits = self.store.search_by_embedding(ent.embedding, top_k=3)
                if hits and hits[0][1] >= self.embedding_threshold:
                    chosen = self.store.find_root(hits[0][0])
                    logger.debug(
                        "Reusing entity node %s via embedding match score=%.3f for aliases=%s",
                        chosen,
                        hits[0][1],
                        aliases,
                    )

            if chosen is None:
                node = self.store.add_node(
                    node_type=ent.type or "entity",
                    aliases=aliases,
                    properties=ent.properties,
                    embedding=ent.embedding,
                    confidence=ent.confidence,
                    provenance={"source": "extraction"},
                    valid_from=turn_id,
                )
                chosen = node.node_id
                logger.debug(
                    "Created new entity node %s for aliases=%s (type=%s, turn=%s)",
                    chosen,
                    aliases,
                    ent.type or "entity",
                    turn_id,
                )
            else:
                node = self.store.nodes[chosen]
                new_aliases = [a for a in aliases if a not in node.aliases]
                if new_aliases:
                    node.aliases.extend(new_aliases)
                    for surface in new_aliases:
                        self.store.surface_index[surface.lower()].add(chosen)
                    logger.debug(
                        "Augmented aliases for entity %s with %s (turn=%s)",
                        chosen,
                        new_aliases,
                        turn_id,
                    )
            mapping[ent.name] = chosen
        return mapping

    def _ground_events(
        self,
        events: List[ExtractedEvent],
        entity_map: Dict[str, str],
        turn_id: Optional[int],
    ) -> Dict[str, str]:
        event_nodes: Dict[str, str] = {}
        for ev in events:
            # Reuse an existing event node if the description already exists as an alias.
            candidates = self.store.lookup_surface(ev.description)
            roots = (
                {self.store.find_root(cid) for cid in candidates}
                if candidates
                else set()
            )
            chosen: Optional[str] = None
            if roots:
                chosen = next(iter(roots))
                for other in list(roots)[1:]:
                    chosen = self.store.merge_aliases(chosen, other)
                logger.debug(
                    "Reusing event node %s for description=%r (roots=%s)",
                    chosen,
                    ev.description,
                    roots,
                )

            if chosen is None:
                event_props = (
                    ev.properties
                    if isinstance(ev.properties, dict)
                    else _normalize_properties(ev.properties)
                )
                node = self.store.add_node(
                    node_type="event",
                    aliases=[ev.description],
                    properties={"description": ev.description, **event_props},
                    confidence=ev.confidence,
                    provenance={"source": "extraction"},
                    valid_from=turn_id if ev.time is None else ev.time,
                )
                chosen = node.node_id
                logger.debug(
                    "Created new event node %s for description=%r (participants=%s, turn=%s)",
                    chosen,
                    ev.description,
                    ev.participants,
                    turn_id if ev.time is None else ev.time,
                )
            else:
                # Ensure the description is captured as an alias on the reused node.
                node = self.store.nodes[chosen]
                if ev.description not in node.aliases:
                    node.aliases.append(ev.description)
                    self.store.surface_index[ev.description.lower()].add(chosen)
                    logger.debug(
                        "Augmented aliases for event %s with %r", chosen, ev.description
                    )

            event_nodes[ev.description] = node.node_id
            for participant in ev.participants:
                src = entity_map.get(participant) or self._ensure_entity(
                    participant, entity_map, turn_id
                )
                existing_edge = next(
                    (
                        e
                        for e in self.store.active_edges()
                        if e.source == src
                        and e.target == node.node_id
                        and e.rel_label == "participates_in"
                    ),
                    None,
                )
                if existing_edge:
                    logger.debug(
                        "Reusing participates_in edge %s from %s to event %s for participant=%r",
                        existing_edge.edge_id,
                        src,
                        node.node_id,
                        participant,
                    )
                    continue
                edge = self.store.add_edge(
                    source=src,
                    target=node.node_id,
                    rel_label="participates_in",
                    provenance={"source": "extraction"},
                    valid_from=turn_id if ev.time is None else ev.time,
                    confidence=ev.confidence,
                )
                logger.debug(
                    "Created participates_in edge %s from %s to event %s for participant=%r (turn=%s)",
                    edge.edge_id,
                    src,
                    node.node_id,
                    participant,
                    turn_id if ev.time is None else ev.time,
                )
        return event_nodes

    def _ground_relations(
        self,
        relations: List[ExtractedRelation],
        entity_map: Dict[str, str],
        turn_id: Optional[int],
    ) -> List[str]:
        edges: List[str] = []
        for rel in relations:
            source_id = entity_map.get(rel.source) or self._ensure_entity(
                rel.source, entity_map, turn_id
            )
            target_id = entity_map.get(rel.target) or self._ensure_entity(
                rel.target, entity_map, turn_id
            )
            edge = self.store.add_edge(
                source=source_id,
                target=target_id,
                rel_label=rel.rel_label,
                confidence=rel.confidence,
                provenance=(
                    {"source": "extraction", "evidence": rel.evidence}
                    if rel.evidence
                    else {"source": "extraction"}
                ),
                valid_from=turn_id if rel.time is None else rel.time,
            )
            edges.append(edge.edge_id)
            logger.debug(
                "Created relation edge %s (%s -> %s) label=%s (turn=%s, conf=%.2f)",
                edge.edge_id,
                source_id,
                target_id,
                rel.rel_label,
                turn_id if rel.time is None else rel.time,
                rel.confidence,
            )
        return edges

    def _ensure_entity(
        self, name: str, entity_map: Dict[str, str], turn_id: Optional[int]
    ) -> str:
        if name in entity_map:
            return entity_map[name]
        node = self.store.add_node(
            node_type="entity",
            aliases=[name],
            provenance={
                "source": "extraction",
                "note": "auto-created for relation grounding",
            },
            valid_from=turn_id,
        )
        entity_map[name] = node.node_id
        logger.debug(
            "Auto-created entity node %s for name=%r while grounding relation/preference (turn=%s)",
            node.node_id,
            name,
            turn_id,
        )
        return node.node_id
