from __future__ import annotations

"""Lightweight memory manager for extracting entities from observations."""

import json
from pathlib import Path
from typing import Optional, Literal, Any, Set

from graphviz import Digraph

from llm import LLM
from memory import Memory


class MemoryManager:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self.memory = Memory()
        self._max_retries = 5
        self._graph = Digraph(comment="Memory Snapshots")
        self._graph_nodes: Set[str] = set()
        self._graph_edges: Set[tuple[str, str, str]] = set()
        self._snapshot_step = 0
        self._graph_dir = Path("out")
        self._graph_dir.mkdir(parents=True, exist_ok=True)
        self._last_entities: list[str] = []

    def _extract_relevant_entities(
        self, observation: str, last_action: str
    ) -> list[str]:
        action_text = last_action or "None"
        prompt = (
            "Extract distinct entities from the action and observation.\n"
            "Rules:\n"
            "- An entity is either an object, location, or person"
            "- Output only unique entities - do NOT repeat entities.\n"
            "- All entities should be in present in the prompt - do NOT make anything up.\n"
            "- Make each entity a concise, descriptive noun phrase so it is recognizable (do not include 'the').\n"
            "- Respond with a list of strings (ex. [answer 1, answer 2, etc.]).\n"
            "- End the response with the stop token <END>.\n\n"
            f"Action: {action_text}\n"
            f"Observation: {observation}\n"
            "Entities (list):"
        )
        completion = self.llm.generate(prompt, max_tokens=96, stop=["<END>"])
        text = completion.split("<END>", 1)[0].strip()
        print(f"{text=}")
        prompt = (
            "Extract the unique entities from the list of raw entities. Note there might be duplicates\n"
            "Rules:\n"
            "- Output only unique entities - do NOT repeat entities.\n"
            "- Only output the entity names as a comma-separated list of strings.\n"
            "- End the response with the stop token <END>.\n\n"
            f"Entities: {completion}\n"
            "Unique Entities (list): "
        )
        completion = self.llm.generate(prompt, max_tokens=96, stop=["<END>"])

        text = completion.split("<END>", 1)[0].strip()
        entities: list[str] = []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                entities = [
                    str(item).replace("[", "").replace("]", "").strip()
                    for item in parsed
                    if str(item).strip()
                ]
        except Exception:
            entities = [
                part.replace("[", "").replace("]", "").strip()
                for part in text.split(",")
                if part.strip()
            ]
        entities = list(set(entities))
        print(f"{entities=}")
        return entities

    def _with_retries(self, fn, desc: str):
        last_exc: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return fn()
            except Exception as exc:  # pragma: no cover - best effort
                last_exc = exc
                print(f"[retry {attempt}/{self._max_retries}] {desc} failed: {exc}")
        if last_exc:
            raise last_exc
        raise RuntimeError(f"Failed to complete {desc}")

    def _classify_entity(
        self, entity: str, observation: str
    ) -> Literal["location", "object"]:
        def classify() -> Literal["location", "object"]:
            prompt = (
                "Respond with the correct type of the entity\n"
                "Respond with exactly one word - location or object.\n\n"
                f"Observation: {observation}"
                f"Based on the observation, out of object and location, the type of {entity} is "
            )
            completion = (
                self.llm.generate(prompt, max_tokens=4, stop=None).strip().lower()
            )
            if "location" in completion:
                return "location"
            if "object" in completion:
                return "object"
            raise Exception(
                f"Error when classifying entity {entity} - received {completion=}"
            )

        return self._with_retries(classify, "classify_entity")

    def _classify_action(
        self, action
    ) -> Literal["navigation", "manipulation", "perception", "start"]:
        if action == "start":
            return "start"

        def classify() -> Literal["navigation", "manipulation", "perception", "start"]:
            prompt = (
                "I want to know the type of the action. Respond with the correct type of the action.\n"
                "There are 3 types: navigation (movement like go north, enter), manipulation (interacting with objects like open door, take key), perception (observing like look, examine, inventory).\n"
                "Respond with exactly one of: navigation, manipulation, perception.\n\n"
                "Make sure that you are confident about your answers"
                "Finish your output with <END>"
                f"Type of action '{action}' is "
            )
            completion = (
                self.llm.generate(prompt, max_tokens=6, stop=["<END>"]).strip().lower()
            )
            if "navigation" in completion:
                return "navigation"
            if "manipulation" in completion:
                return "manipulation"
            if "perception" in completion:
                return "perception"
            raise Exception(
                f"Error when classifying action '{action}' - received {completion=}"
            )

        return self._with_retries(classify, "classify_action")

    def _infer_entity_attributes(
        self, observation: str, action: str, entity: str, attributes: list[str]
    ) -> dict[str, Any]:
        def infer() -> dict[str, Any]:
            attrs = ", ".join(attributes) if attributes else "none"
            prompt = (
                "You are extracting the state of an entity in a text adventure.\n"
                "Given the last action, observation, and entity, return a JSON object of attribute -> value pairs.\n"
                "ONLY come up with attributes based on the given information - do NOT make anything up.\n"
                "Use booleans for on/off, open/closed, locked/unlocked when appropriate. Use numbers or short strings otherwise. If unknown, use null.\n"
                "End the response with <END> immediately after the JSON.\n\n"
                f"Entity: {entity}\n"
                f"Known attributes: {attrs}\n"
                f"Last action: {action}\n"
                f"Observation: {observation}\n"
                "Attributes JSON:"
            )
            completion = self.llm.generate(
                prompt, max_tokens=128, stop=["<END>"]
            ).strip()
            text = completion.split("<END>", 1)[0].strip()
            parsed: dict[str, Any] = {}
            try:
                parsed_obj = json.loads(text)
                if isinstance(parsed_obj, dict):
                    parsed = parsed_obj
            except Exception:
                pass
            if not parsed:
                raise Exception("Could not parse attributes JSON")
            return parsed

        return self._with_retries(infer, "infer_entity_attributes")

    def _update_location(self, observation: str, navigation_action: str) -> None:
        """
        Use the LLM to infer the current location from the observation, update the
        player's location, and add a directional edge from the previous location.
        """

        def locate() -> str:
            prompt = (
                "You are playing a text adventure game."
                "Identify your current location from this observation.\n"
                "Only respond with the location name (short word or phrase, normally capitalized), and then the <END> token.\n"
                f"Observation: {observation}\n"
                "Based on the given observation, my location is: "
            )
            completion = self.llm.generate(prompt, max_tokens=6, stop=["<END>"]).strip()
            if not completion:
                raise Exception("Unable to extract location from observation")
            return completion

        location_name = self._with_retries(locate, "update_location")
        prev_loc = str(self.memory.player.get("location") or "")
        self.memory.add_location_node(location_name)
        self.memory.set_player_location(location_name)
        self.memory.add_location_edge(prev_loc, navigation_action, location_name)
        print(f"UPDATED location from '{prev_loc}' to '{location_name}'")

    def _add_snapshot_transition(
        self, prev_hash: str, curr_hash: str, action: str
    ) -> None:
        """Add nodes/edge to the snapshot graph (allows self-loops)."""
        if prev_hash not in self._graph_nodes:
            self._graph.node(prev_hash, label=prev_hash)
            self._graph_nodes.add(prev_hash)
        if curr_hash not in self._graph_nodes:
            self._graph.node(curr_hash, label=curr_hash)
            self._graph_nodes.add(curr_hash)
        edge_label = action or "None"
        edge_key = (prev_hash, edge_label, curr_hash)
        if edge_key in self._graph_edges:
            return
        self._graph_edges.add(edge_key)
        self._graph.edge(
            prev_hash,
            curr_hash,
            label=edge_label,
            _attributes={"id": f"{prev_hash}->{curr_hash}:{edge_label}"},
        )

    def _render_graph(self, step_index: int) -> None:
        """Render the current graph snapshot for the given step index."""
        filename = self._graph_dir / f"memory_graph_step_{step_index}"
        try:
            self._graph.render(
                filename=str(filename),
                format="png",
                cleanup=True,
                view=False,
            )
        except Exception as exc:  # pragma: no cover - best effort logging
            print(f"Graph render failed at step {step_index}: {exc}")

    def _export_state_components(self, step_index: int) -> None:
        """Export locations graph (graphviz) plus player/objects JSON."""
        snapshot = self.memory.to_dict()

        # Locations graph via Graphviz
        loc_graph = Digraph(comment=f"Locations step {step_index}")
        for loc in snapshot.get("locations", {}):
            loc_graph.node(loc, label=loc)
        for src, neighbors in snapshot.get("locations", {}).items():
            if isinstance(neighbors, dict):
                for direction, dest in neighbors.items():
                    loc_graph.edge(src, dest, label=direction)
        try:
            loc_graph.render(
                filename=str(self._graph_dir / f"locations_step_{step_index}"),
                format="png",
                cleanup=True,
                view=False,
            )
        except Exception as exc:  # pragma: no cover - best effort
            print(f"Location graph render failed at step {step_index}: {exc}")

        # Player and objects as JSON
        (self._graph_dir / f"player_step_{step_index}.json").write_text(
            json.dumps(snapshot.get("player", {}), indent=2)
        )
        (self._graph_dir / f"objects_step_{step_index}.json").write_text(
            json.dumps(snapshot.get("objects", {}), indent=2)
        )

    def update_memory(self, observation: str, last_action: str) -> None:
        print("=" * 20)
        print("Updating memory")
        prev_hash = str(hash(self.memory))
        action_type = self._classify_action(last_action)
        print(f"action '{last_action}' is of type {action_type}")
        entities = self._extract_relevant_entities(observation, last_action)
        self._last_entities = entities
        num_locations = 0
        for entity in entities:
            if entity in self.memory.entity_map:
                entity_type = self.memory.entity_map[entity]
            else:
                # add entity to memory
                entity_type = self._classify_entity(entity, observation)
                if entity_type == "object":
                    self.memory.add_object(entity)
                elif entity_type == "location":
                    self.memory.add_location_node(entity)
            if entity_type == "object":
                current_attrs = list(
                    (self.memory.get_object_state(entity) or {}).keys()
                )
                updates = self._infer_entity_attributes(
                    observation, last_action, entity, current_attrs
                )
                print(
                    f"entity '{entity}' information: {self.memory.objects.get(entity, {})}"
                )

                if updates:
                    if self.memory.set_object_state(entity, updates):
                        print(f"UPDATES: {updates}")
            print(f"entity '{entity}' is of type {entity_type}")
            if entity_type == "location":
                num_locations += 1

        if action_type in {"navigation", "start"} and num_locations > 0:
            self._update_location(observation, last_action)

        curr_hash = str(hash(self.memory))
        self._add_snapshot_transition(prev_hash, curr_hash, last_action)
        self._render_graph(self._snapshot_step)
        self._export_state_components(self._snapshot_step)
        self._snapshot_step += 1

        print("SNAPSHOT:")
        print(self.memory._snapshot())
        print("=" * 20)
        print()

    def get_recent_entities_context(self) -> list[dict[str, Any]]:
        """Return context for the most recently extracted entities."""
        context: list[dict[str, Any]] = []
        for ent in self._last_entities:
            entry: dict[str, Any] = {"name": ent}
            ent_type = self.memory.entity_map.get(ent)
            if ent_type:
                entry["type"] = ent_type
                if ent_type == "object":
                    entry["state"] = self.memory.get_object_state(ent) or {}
                if ent_type == "location":
                    entry["neighbors"] = self.memory.neighbors(ent)
            context.append(entry)
        return context


__all__ = ["MemoryManager"]
