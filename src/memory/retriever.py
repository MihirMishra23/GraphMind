from __future__ import annotations

from typing import Iterable, Set

from .schema import EdgeType, NodeType, WorldKG


class KGRetriever:
    def __init__(self, world_kg: WorldKG):
        self.world_kg = world_kg

    def find_relevant_entities(self, obs: str, step: int) -> list[str]:
        """
        Heuristics:
        - Start from entities explicitly mentioned in obs (string match).
        - Always include PLAYER, current room, inventory items.
        - Optionally include all NOTES and FLAGS.
        """
        obs_lower = obs.lower()
        relevant: Set[str] = set()

        # Always include player.
        player_ids = [n for n, data in self.world_kg.graph.nodes(data=True) if data.get("type") == NodeType.PLAYER.value]
        relevant.update(player_ids)

        # Include player's location from state.
        for pid in player_ids:
            location = self.world_kg.graph.nodes[pid].get("state", {}).get("location")
            if location and location in self.world_kg.graph:
                relevant.add(location)

        # Inventory items (HAS edges from player).
        for pid in player_ids:
            for _, obj, data in self.world_kg.graph.edges(pid, data=True):
                if data.get("type") == EdgeType.HAS.value:
                    relevant.add(obj)

        # Mentions in obs by name or aliases.
        relevant.update(self._find_mentions(obs_lower))

        # Optionally include notes and flags (helpful context).
        for node_id, data in self.world_kg.graph.nodes(data=True):
            if data.get("type") in {NodeType.NOTE.value, NodeType.FLAG.value}:
                relevant.add(node_id)

        # Ensure current room containment brings its contents nearby.
        for node_id in list(relevant):
            for src, dst, data in self.world_kg.graph.edges(node_id, data=True):
                if data.get("type") == EdgeType.IN.value:
                    relevant.add(src)
            for src, dst, data in self.world_kg.graph.in_edges(node_id, data=True):
                if data.get("type") == EdgeType.IN.value:
                    relevant.add(src)

        return list(relevant)

    def get_relevant_subgraph(self, obs: str, step: int, radius: int = 2) -> WorldKG:
        """
        Use find_relevant_entities + world_kg.get_local_subgraph.
        """
        centers = self.find_relevant_entities(obs, step)
        if not centers:
            centers = list(self.world_kg.graph.nodes)[:1]  # fallback to some node if graph not empty
        return self.world_kg.get_local_subgraph(centers, radius=radius)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _find_mentions(self, obs_lower: str) -> Set[str]:
        found: Set[str] = set()
        for node_id, data in self.world_kg.graph.nodes(data=True):
            names: Iterable[str] = [data.get("name", node_id)]
            aliases = data.get("aliases") or []
            names = list(names) + list(aliases)
            for name in names:
                if not name:
                    continue
                if name.lower() in obs_lower:
                    found.add(node_id)
                    break
        return found
