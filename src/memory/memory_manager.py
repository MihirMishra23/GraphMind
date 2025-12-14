from __future__ import annotations

"""Lightweight memory manager for extracting entities from observations."""

import json
from typing import Optional, Literal, Any

from llm import LLM
from memory import Memory


class MemoryManager:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self.memory = Memory()
        self._max_retries = 3

    def _extract_relevant_entities(
        self, observation: str, last_action: str
    ) -> list[str]:
        action_text = last_action or "None"
        prompt = (
            "Extract distinct entities (objects, locations, items) from the last action and observation.\n"
            "Rules:\n"
            "- Output only unique entities; never repeat the same entity.\n"
            "- Make each entity a concise, descriptive noun phrase so it is recognizable, but keep it short.\n"
            "- Respond with a list of strings.\n"
            "- End the response with the stop token <END> immediately after the JSON.\n\n"
            "Example 1:\n"
            "Last action: look at the wooden chest\n"
            "Observation: The wooden chest is open, revealing a rusty iron key inside.\n"
            'Entities (list): ["wooden chest", "rusty iron key"]<END>\n\n'
            "Example 2:\n"
            "Last action: go north\n"
            "Observation: You enter a stone hallway lined with oil lamps and see a locked oak door ahead.\n"
            'Entities (list): ["stone hallway", "oil lamps", "locked oak door"]<END>\n'
            f"Last action: {action_text}\n"
            f"Observation: {observation}\n"
            "Entities (list):"
        )
        completion = self.llm.generate(prompt, max_tokens=96, stop=["<END>"])
        text = completion.split("<END>", 1)[0].strip()
        entities: list[str] = []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                entities = [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            entities = [part.strip() for part in text.split(",") if part.strip()]
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

    def _classify_entity(self, entity) -> Literal["location", "object"]:
        def classify() -> Literal["location", "object"]:
            prompt = (
                "I want to know the type of the entity. Respond with the correct type of the entity\n"
                "Respond with exactly one word - location or object.\n\n"
                # "Finish your output with <END>."
                f"Type of {entity}: "
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
                "Finish your output with <END>"
                "Example 1:\n"
                "Type of action 'go north' is navigation<END>\n\n"
                "Example 2:\n"
                "Type of action 'open the mailbox' is manipulation<END>\n\n"
                "Example 3:\n"
                "Type of action 'look around' is perception<END>\n\n"
                "Example 4:\n"
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
                "You are inferring the state of an entity in a text adventure.\n"
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
            print(f"Inferred attributes: {completion}")
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
            print(f"{prompt=}")
            completion = self.llm.generate(prompt, max_tokens=6, stop=["<END>"]).strip()
            print(f"{completion=}")
            if not completion:
                raise Exception("Unable to extract location from observation")
            return completion

        location_name = self._with_retries(locate, "update_location")
        prev_loc = str(self.memory.player.get("location") or "")
        self.memory.add_location_node(location_name)
        self.memory.set_player_location(location_name)
        print(f"{prev_loc=}")
        self.memory.add_location_edge(prev_loc, navigation_action, location_name)

    def update_memory(self, observation: str, last_action: str) -> None:
        print("=" * 20)
        print("Updating memory")
        action_type = self._classify_action(last_action)
        print(f"action '{last_action}' is of type {action_type}")
        entities = self._extract_relevant_entities(observation, last_action)
        num_locations = 0
        for entity in entities:
            if entity in self.memory.entity_map:
                entity_type = self.memory.entity_map[entity]
            else:
                # add entity to memory
                entity_type = self._classify_entity(entity)
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
                if updates:
                    self.memory.set_object_state(entity, updates)
            print(f"entity '{entity}' is of type {entity_type}")
            if entity_type == "location":
                num_locations += 1

        if action_type in {"navigation", "start"} and num_locations > 0:
            self._update_location(observation, last_action)

        print(self.memory._snapshot())
        print("=" * 20)
        print()


__all__ = ["MemoryManager"]
