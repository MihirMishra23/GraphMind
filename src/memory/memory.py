from __future__ import annotations

import json
from typing import Optional


class Memory:
    """Holds location graph, object state, and player state."""

    def __init__(self) -> None:
        self.locations: dict[str, dict[str, str]] = {"start": {}}
        self.objects: dict[str, dict] = {}
        self.player: dict[str, object] = {"location": "start", "inventory": []}

    def _normalize(self, value: Optional[str], field: str) -> str:
        if value is None:
            raise ValueError(f"{field} cannot be None")
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError(f"{field} cannot be empty")
        return normalized

    def add_location_node(self, location_id: str) -> None:
        """Ensure a location exists."""
        loc = self._normalize(location_id, "location_id")
        self.locations.setdefault(loc, {})

    def add_location_edge(self, src: str, direction: str, dest: str) -> None:
        """Add a directional edge from src to dest labeled by direction."""
        src_id = self._normalize(src, "src")
        dest_id = self._normalize(dest, "dest")
        dir_id = self._normalize(direction, "direction")
        assert (
            src_id in self.locations and dest_id in self.locations
        ), "Both locations must exist before adding an edge"
        self.locations[src_id][dir_id] = dest_id

    def neighbors(self, location_id: str) -> dict[str, str]:
        """Return neighbors (direction -> destination) for a location."""
        loc = self._normalize(location_id, "location_id")
        return self.locations.get(loc, {})

    def list_locations(self) -> list[str]:
        """List known locations."""
        return list(self.locations.keys())

    def set_object_state(self, object_id: str, state: dict) -> None:
        """Merge provided attributes into an object's state."""
        obj = self._normalize(object_id, "object_id")
        if not isinstance(state, dict):
            raise TypeError("state must be a dict")
        current = self.objects.setdefault(obj, {})
        current.update(state)

    def get_object_state(self, object_id: str) -> Optional[dict]:
        """Get state for an object."""
        obj = self._normalize(object_id, "object_id")
        return self.objects.get(obj)

    def list_objects(self) -> list[str]:
        """List known objects."""
        return list(self.objects.keys())

    def set_player_location(self, location_id: str) -> None:
        """Set the player's current location."""
        loc = self._normalize(location_id, "location_id")
        self.add_location_node(loc)
        self.player["location"] = loc

    def add_to_inventory(self, object_id: str) -> None:
        """Add an object to the player's inventory."""
        obj = self._normalize(object_id, "object_id")
        inventory: list[str] = self.player["inventory"]  # type: ignore[assignment]
        if obj not in inventory:
            inventory.append(obj)

    def remove_from_inventory(self, object_id: str) -> None:
        """Remove an object from the player's inventory."""
        obj = self._normalize(object_id, "object_id")
        inventory: list[str] = self.player["inventory"]  # type: ignore[assignment]
        if obj in inventory:
            inventory.remove(obj)

    def list_inventory(self) -> list[str]:
        """List objects in the player's inventory."""
        inventory: list[str] = self.player["inventory"]  # type: ignore[assignment]
        return list(inventory)

    def to_dict(self) -> dict:
        """Return a serializable snapshot."""
        return {
            "locations": self.locations,
            "objects": self.objects,
            "player": {
                "location": self.player["location"],
                "inventory": list(self.player["inventory"]),  # type: ignore[index]
            },
        }

    def to_json(self) -> str:
        """Serialize memory to JSON."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        """Reconstruct Memory from a dict snapshot."""
        mem = cls()
        for loc, neighbors in data.get("locations", {}).items():
            mem.add_location_node(loc)
            if isinstance(neighbors, dict):
                for direction, dest in neighbors.items():
                    mem.add_location_edge(loc, direction, dest)
        for obj_id, state in data.get("objects", {}).items():
            if isinstance(state, dict):
                mem.set_object_state(obj_id, state)
        player = data.get("player", {}) or {}
        if isinstance(player, dict):
            loc = player.get("location")
            if loc:
                mem.set_player_location(loc)
            for item in player.get("inventory", []) or []:
                mem.add_to_inventory(item)
        return mem

    @classmethod
    def from_json(cls, json_str: str) -> "Memory":
        """Reconstruct Memory from JSON."""
        return cls.from_dict(json.loads(json_str))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Memory):
            return False
        return self._snapshot() == other._snapshot()

    def __hash__(self) -> int:  # pragma: no cover - deterministic via snapshot
        return hash(self._snapshot())

    def _snapshot(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


__all__ = ["Memory"]
