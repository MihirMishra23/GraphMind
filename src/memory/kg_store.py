from __future__ import annotations

import copy

from .schema import WorldKG


class KGSnapshots:
    def __init__(self):
        self.snapshots: dict[int, WorldKG] = {}

    def store_snapshot(self, step: int, kg: WorldKG) -> None:
        """Store a deep copy of the world KG for this step."""
        self.snapshots[step] = self._copy_world_kg(kg)

    def get_snapshot(self, step: int) -> WorldKG:
        """Retrieve snapshot (for evaluation or replay)."""
        if step not in self.snapshots:
            raise KeyError(f"No snapshot stored for step {step}")
        return self._copy_world_kg(self.snapshots[step])

    def _copy_world_kg(self, kg: WorldKG) -> WorldKG:
        clone = WorldKG()
        clone.graph = copy.deepcopy(kg.graph)
        return clone
