"""Property graph memory implementation with open-vocab relations and validity tracking."""
from __future__ import annotations

import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class Node:
    node_id: str
    node_type: str
    aliases: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    confidence: float = 1.0
    provenance: Dict[str, Any] = field(default_factory=dict)
    valid_from: Optional[int] = None
    valid_to: Optional[int] = None
    evidence_count: int = 1
    contradicted: bool = False
    last_updated: int = 0


@dataclass
class Edge:
    edge_id: str
    source: str
    target: str
    rel_label: str
    rel_embedding: Optional[List[float]] = None
    rel_cluster_id: Optional[str] = None
    valid_from: Optional[int] = None
    valid_to: Optional[int] = None
    confidence: float = 1.0
    evidence_count: int = 1
    provenance: Dict[str, Any] = field(default_factory=dict)
    contradicted: bool = False
    last_updated: int = 0


class GraphStore:
    """In-memory property graph with aliasing, provenance, and interval semantics."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self.alias_parent: Dict[str, str] = {}
        self.surface_index: Dict[str, Set[str]] = defaultdict(set)
        self.embedding_index: List[Tuple[str, List[float]]] = []
        self.recency_index: List[Tuple[int, str, str]] = []
        self._clock = 0

    # Basic utilities -----------------------------------------------------
    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _next_node_id(self) -> str:
        return f"n-{uuid.uuid4().hex}"

    def _next_edge_id(self) -> str:
        return f"e-{uuid.uuid4().hex}"

    def _record_recency(self, kind: str, obj_id: str) -> None:
        self.recency_index.append((self._clock, kind, obj_id))

    # Node operations -----------------------------------------------------
    def add_node(
        self,
        node_type: str,
        aliases: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        confidence: float = 1.0,
        provenance: Optional[Dict[str, Any]] = None,
        valid_from: Optional[int] = None,
        evidence_count: int = 1,
    ) -> Node:
        ts = self._tick()
        node_id = self._next_node_id()
        node = Node(
            node_id=node_id,
            node_type=node_type,
            aliases=list(aliases or []),
            properties=properties or {},
            embedding=embedding,
            confidence=confidence,
            provenance=provenance or {},
            valid_from=valid_from or ts,
            valid_to=None,
            evidence_count=evidence_count,
            last_updated=ts,
        )
        self.nodes[node_id] = node
        self.alias_parent[node_id] = node_id
        for surface in node.aliases:
            self.surface_index[surface.lower()].add(node_id)
        if embedding is not None:
            self.embedding_index.append((node_id, embedding))
        self._record_recency("node", node_id)
        return node

    def delete_node(self, node_id: str, contradicted: bool = False, at: Optional[int] = None) -> None:
        node = self.nodes.get(node_id)
        if node is None:
            return
        ts = at or self._tick()
        if node.valid_to is None or node.valid_to > ts:
            node.valid_to = ts
        node.contradicted = contradicted
        node.last_updated = ts
        self._record_recency("node", node_id)

    # Edge operations -----------------------------------------------------
    def add_edge(
        self,
        source: str,
        target: str,
        rel_label: str,
        rel_embedding: Optional[List[float]] = None,
        rel_cluster_id: Optional[str] = None,
        confidence: float = 1.0,
        provenance: Optional[Dict[str, Any]] = None,
        valid_from: Optional[int] = None,
        evidence_count: int = 1,
    ) -> Edge:
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("Both source and target must exist before adding an edge.")
        ts = self._tick()
        edge_id = self._next_edge_id()
        edge = Edge(
            edge_id=edge_id,
            source=source,
            target=target,
            rel_label=rel_label,
            rel_embedding=rel_embedding,
            rel_cluster_id=rel_cluster_id,
            valid_from=valid_from or ts,
            valid_to=None,
            confidence=confidence,
            provenance=provenance or {},
            evidence_count=evidence_count,
            last_updated=ts,
        )
        self.edges[edge_id] = edge
        self._record_recency("edge", edge_id)
        return edge

    def update_edge(
        self,
        edge_id: str,
        rel_label: Optional[str] = None,
        rel_embedding: Optional[List[float]] = None,
        rel_cluster_id: Optional[str] = None,
        confidence: Optional[float] = None,
        provenance: Optional[Dict[str, Any]] = None,
        evidence_delta: int = 0,
        valid_from: Optional[int] = None,
    ) -> Edge:
        edge = self.edges.get(edge_id)
        if edge is None:
            raise KeyError(f"Edge {edge_id} not found.")
        close_ts = self._tick()
        if edge.valid_to is None or edge.valid_to > close_ts:
            edge.valid_to = close_ts
        edge.last_updated = close_ts
        new_edge = Edge(
            edge_id=self._next_edge_id(),
            source=edge.source,
            target=edge.target,
            rel_label=rel_label or edge.rel_label,
            rel_embedding=edge.rel_embedding if rel_embedding is None else rel_embedding,
            rel_cluster_id=edge.rel_cluster_id if rel_cluster_id is None else rel_cluster_id,
            valid_from=valid_from or close_ts,
            valid_to=None,
            confidence=edge.confidence if confidence is None else confidence,
            provenance={**edge.provenance, **(provenance or {})},
            evidence_count=max(1, edge.evidence_count + evidence_delta),
            last_updated=close_ts,
        )
        self.edges[new_edge.edge_id] = new_edge
        self._record_recency("edge", new_edge.edge_id)
        return new_edge

    def delete_edge(self, edge_id: str, contradicted: bool = False, at: Optional[int] = None) -> None:
        edge = self.edges.get(edge_id)
        if edge is None:
            return
        ts = at or self._tick()
        if edge.valid_to is None or edge.valid_to > ts:
            edge.valid_to = ts
        edge.contradicted = contradicted
        edge.last_updated = ts
        self._record_recency("edge", edge_id)

    # Alias management ----------------------------------------------------
    def find_root(self, node_id: str) -> str:
        parent = self.alias_parent.get(node_id)
        if parent is None:
            raise KeyError(f"Unknown node_id {node_id}")
        if parent != node_id:
            self.alias_parent[node_id] = self.find_root(parent)
        return self.alias_parent[node_id]

    def merge_aliases(self, primary: str, duplicate: str) -> str:
        root_primary = self.find_root(primary)
        root_duplicate = self.find_root(duplicate)
        if root_primary == root_duplicate:
            return root_primary
        self.alias_parent[root_duplicate] = root_primary
        main_node = self.nodes[root_primary]
        dup_node = self.nodes[root_duplicate]
        merged_aliases = set(main_node.aliases) | set(dup_node.aliases)
        main_node.aliases = list(merged_aliases)
        main_node.evidence_count += dup_node.evidence_count
        for surface in merged_aliases:
            self.surface_index[surface.lower()].add(root_primary)
        self._record_recency("node", root_primary)
        self.add_edge(
            source=root_duplicate,
            target=root_primary,
            rel_label="alias_of",
            provenance={"system": "alias_union"},
        )
        return root_primary

    # Lookup helpers ------------------------------------------------------
    def lookup_surface(self, surface: str) -> List[str]:
        return list(self.surface_index.get(surface.lower(), []))

    def search_by_embedding(self, query: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        if not query:
            return []
        results: List[Tuple[str, float]] = []
        for node_id, emb in self.embedding_index:
            if emb is None or len(emb) != len(query):
                continue
            score = self._cosine_similarity(query, emb)
            results.append((node_id, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def recent(self, limit: int = 10, kind: Optional[str] = None) -> List[Tuple[int, str, str]]:
        filtered = self.recency_index if kind is None else [r for r in self.recency_index if r[1] == kind]
        return list(sorted(filtered, key=lambda x: x[0], reverse=True))[:limit]

    # Views ---------------------------------------------------------------
    def active_nodes(self, at: Optional[int] = None) -> List[Node]:
        ts = at or self._clock
        return [n for n in self.nodes.values() if self._is_active(n.valid_from, n.valid_to, ts)]

    def active_edges(self, at: Optional[int] = None) -> List[Edge]:
        ts = at or self._clock
        return [e for e in self.edges.values() if self._is_active(e.valid_from, e.valid_to, ts)]

    def history(self) -> Dict[str, Dict[str, Any]]:
        return {
            "nodes": {node_id: node for node_id, node in self.nodes.items()},
            "edges": {edge_id: edge for edge_id, edge in self.edges.items()},
        }

    # Internal helpers ----------------------------------------------------
    @staticmethod
    def _is_active(valid_from: Optional[int], valid_to: Optional[int], ts: int) -> bool:
        lower_ok = valid_from is None or valid_from <= ts
        upper_ok = valid_to is None or ts < valid_to
        return lower_ok and upper_ok

    @staticmethod
    def _cosine_similarity(x: List[float], y: List[float]) -> float:
        denom = math.sqrt(sum(a * a for a in x)) * math.sqrt(sum(b * b for b in y))
        if denom == 0:
            return 0.0
        return sum(a * b for a, b in zip(x, y)) / denom
