from __future__ import annotations

from enum import Enum
from typing import Any, Iterable

import networkx as nx


class NodeType(str, Enum):
    PLAYER = "PLAYER"
    ROOM = "ROOM"
    OBJECT = "OBJECT"
    CHARACTER = "CHARACTER"
    NOTE = "NOTE"
    FLAG = "FLAG"
    OBSERVATION = "OBSERVATION"


class EdgeType(str, Enum):
    CONNECTED_TO = "CONNECTED_TO"
    IN = "IN"
    HAS = "HAS"
    ON = "ON"
    STATE = "STATE"
    MENTIONS = "MENTIONS"
    ACTION = "ACTION"
    CONTAINS = "CONTAINS"


DIRECTIONS = {"north", "south", "east", "west", "up", "down"}


class WorldKG:
    """
    Thin wrapper over a MultiDiGraph that validates the schema and provides
    utility helpers for common operations.
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()

    # ------------------------------------------------------------------ #
    # Node helpers
    # ------------------------------------------------------------------ #
    def _normalize_node_type(self, node_type: str) -> NodeType:
        try:
            return NodeType(node_type)
        except ValueError as exc:
            raise ValueError(f"Invalid node type '{node_type}'") from exc

    def _normalize_edge_type(self, edge_type: str) -> EdgeType:
        try:
            return EdgeType(edge_type)
        except ValueError as exc:
            raise ValueError(f"Invalid edge type '{edge_type}'") from exc

    def _get_node_type(self, node_id: str) -> NodeType:
        if node_id not in self.graph:
            raise KeyError(f"Unknown node '{node_id}'")
        return self._normalize_node_type(self.graph.nodes[node_id]["type"])

    def add_or_get_entity(
        self, canonical_id: str, node_type: str, name: str | None = None
    ) -> str:
        """Create node if missing, or return existing node id."""
        normalized_type = self._normalize_node_type(node_type)
        if canonical_id in self.graph:
            existing_type = self._normalize_node_type(self.graph.nodes[canonical_id]["type"])
            if existing_type != normalized_type:
                raise ValueError(
                    f"Node '{canonical_id}' already exists with type '{existing_type.value}', "
                    f"cannot change to '{normalized_type.value}'."
                )
            if name and not self.graph.nodes[canonical_id].get("name"):
                self.graph.nodes[canonical_id]["name"] = name
            return canonical_id

        self.graph.add_node(
            canonical_id,
            type=normalized_type.value,
            name=name or canonical_id,
            description="",
            state={},
            last_updated_step=0,
            aliases=[],
        )
        return canonical_id

    # ------------------------------------------------------------------ #
    # Attribute/state helpers
    # ------------------------------------------------------------------ #
    def set_state(self, entity_id: str, key: str, value: Any, step: int) -> None:
        """Set/overwrite a state attribute for an entity."""
        if entity_id not in self.graph:
            raise KeyError(f"Unknown node '{entity_id}'")
        node_data = self.graph.nodes[entity_id]
        if "state" not in node_data or not isinstance(node_data["state"], dict):
            node_data["state"] = {}
        node_data["state"][key] = value
        node_data["last_updated_step"] = step

    # ------------------------------------------------------------------ #
    # Edge helpers
    # ------------------------------------------------------------------ #
    def _validate_edge_types(self, src_type: NodeType, dst_type: NodeType, edge_type: EdgeType) -> None:
        if edge_type == EdgeType.CONNECTED_TO:
            if not (src_type == NodeType.ROOM and dst_type == NodeType.ROOM):
                raise ValueError("CONNECTED_TO edges must link ROOM -> ROOM.")
        elif edge_type == EdgeType.IN:
            if src_type not in {NodeType.OBJECT, NodeType.CHARACTER}:
                raise ValueError("IN edges must start from OBJECT or CHARACTER.")
            if dst_type not in {NodeType.ROOM, NodeType.OBJECT}:
                raise ValueError("IN edges must point to ROOM or OBJECT.")
        elif edge_type == EdgeType.HAS:
            if src_type not in {NodeType.PLAYER, NodeType.CHARACTER} or dst_type != NodeType.OBJECT:
                raise ValueError("HAS edges must be PLAYER/CHARACTER -> OBJECT.")
        elif edge_type == EdgeType.ON:
            if not (src_type == NodeType.OBJECT and dst_type == NodeType.OBJECT):
                raise ValueError("ON edges must link OBJECT -> OBJECT.")
        elif edge_type == EdgeType.STATE:
            if dst_type != NodeType.FLAG:
                raise ValueError("STATE edges must target a FLAG node.")
        elif edge_type == EdgeType.MENTIONS:
            if src_type != NodeType.NOTE:
                raise ValueError("MENTIONS edges must originate from a NOTE.")
        elif edge_type == EdgeType.ACTION:
            if not (src_type == NodeType.OBSERVATION and dst_type == NodeType.OBSERVATION):
                raise ValueError("ACTION edges must link OBSERVATION -> OBSERVATION.")
        elif edge_type == EdgeType.CONTAINS:
            # Allow observation->entity or entity->entity containment.
            if src_type not in {NodeType.OBSERVATION, NodeType.PLAYER, NodeType.ROOM, NodeType.OBJECT, NodeType.CHARACTER, NodeType.NOTE, NodeType.FLAG}:
                raise ValueError("CONTAINS edges must originate from OBSERVATION or ENTITY nodes.")
            if dst_type == NodeType.OBSERVATION:
                raise ValueError("CONTAINS edges must target an entity node.")

    def add_relation(
        self, src_id: str, dst_id: str, rel_type: str, **attrs: Any
    ) -> None:
        """Add a typed directed edge with attributes."""
        rel_enum = self._normalize_edge_type(rel_type)
        src_type = self._get_node_type(src_id)
        dst_type = self._get_node_type(dst_id)
        self._validate_edge_types(src_type, dst_type, rel_enum)

        if rel_enum == EdgeType.CONNECTED_TO:
            direction = attrs.get("direction")
            if direction is None or direction not in DIRECTIONS:
                raise ValueError(
                    f"CONNECTED_TO edges require a valid direction ({', '.join(sorted(DIRECTIONS))})."
                )
        # For ACTION edges, store the action command as the edge label/name.
        if rel_enum == EdgeType.ACTION:
            command = attrs.get("command")
            if not command:
                raise ValueError("ACTION edges require a 'command' attribute.")
            attrs = dict(attrs)
            attrs["name"] = command

        attrs = dict(attrs)
        attrs["type"] = rel_enum.value
        self.graph.add_edge(src_id, dst_id, **attrs)

    def move_entity(self, entity_id: str, new_container_id: str, step: int) -> None:
        """Ensure entity is IN exactly one container (room or object)."""
        entity_type = self._get_node_type(entity_id)
        container_type = self._get_node_type(new_container_id)
        if entity_type not in {NodeType.OBJECT, NodeType.CHARACTER}:
            raise ValueError("Only OBJECT or CHARACTER nodes can be moved.")
        if container_type not in {NodeType.ROOM, NodeType.OBJECT}:
            raise ValueError("Entities can only be moved into a ROOM or OBJECT.")

        # Remove old IN edges originating from this entity.
        for _, dst, key, data in list(self.graph.out_edges(entity_id, keys=True, data=True)):
            if data.get("type") == EdgeType.IN.value:
                self.graph.remove_edge(entity_id, dst, key=key)

        self.graph.nodes[entity_id]["last_updated_step"] = step
        self.add_relation(entity_id, new_container_id, EdgeType.IN.value, last_updated_step=step)

    # ------------------------------------------------------------------ #
    # Query / serialization helpers
    # ------------------------------------------------------------------ #
    def _collect_nodes_within_radius(self, centers: Iterable[str], radius: int) -> set[str]:
        nodes: set[str] = set()
        for center in centers:
            if center not in self.graph:
                continue
            ego = nx.ego_graph(self.graph, center, radius=radius, undirected=True)
            nodes.update(ego.nodes)
        return nodes

    def get_local_subgraph(
        self, center_entities: list[str], radius: int = 2
    ) -> WorldKG:
        """
        Return a new WorldKG instance representing a k-hop neighborhood around
        the given entities.
        """
        nodes_to_keep = self._collect_nodes_within_radius(center_entities, radius)
        subgraph = self.graph.subgraph(nodes_to_keep).copy()

        sub_kg = WorldKG()
        sub_kg.graph = subgraph
        return sub_kg

    def _format_state(self, state: dict[str, Any]) -> str:
        if not state:
            return ""
        return "; ".join(f"{k}={v}" for k, v in state.items())

    def to_text_summary(self, entities: list[str] | None = None) -> str:
        """
        Serialize selected part of the graph into a concise textual description
        suitable for passing to an LLM.
        """
        if entities is None:
            nodes = list(self.graph.nodes)
        else:
            nodes = [n for n in entities if n in self.graph]

        lines: list[str] = []

        # Describe nodes
        for node_id in sorted(nodes):
            data = self.graph.nodes[node_id]
            state_txt = self._format_state(data.get("state", {}))
            desc_parts = [f"[{data['type']}] {data.get('name', node_id)}"]
            if data.get("description"):
                desc_parts.append(f"desc: {data['description']}")
            if state_txt:
                desc_parts.append(f"state: {state_txt}")
            lines.append(f"{node_id}: " + " | ".join(desc_parts))

        # Describe edges between included nodes
        for src, dst, data in self.graph.edges(nodes, data=True):
            if src not in nodes or dst not in nodes:
                continue
            rel = data.get("type", "")
            if rel == EdgeType.CONNECTED_TO.value:
                direction = data.get("direction", "?")
                lines.append(f"{src} -[{direction}]-> {dst}")
            elif rel == EdgeType.IN.value:
                lines.append(f"{src} IN {dst}")
            elif rel == EdgeType.HAS.value:
                lines.append(f"{src} HAS {dst}")
            elif rel == EdgeType.ON.value:
                lines.append(f"{src} ON {dst}")
            elif rel == EdgeType.STATE.value:
                lines.append(f"{src} STATE {dst}")
            elif rel == EdgeType.MENTIONS.value:
                lines.append(f"{src} MENTIONS {dst}")
            else:
                lines.append(f"{src} -> {dst} ({rel})")

        return "\n".join(lines)
