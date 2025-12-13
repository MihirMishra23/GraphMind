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
                "Finish your output with <END>."
                f"Type of {entity} is "
            )
            completion = (
                self.llm.generate(prompt, max_tokens=4, stop=["<END>"]).strip().lower()
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

    def _update_location(
        self, observation: str, navigation_action: str, potential_locations: list
    ) -> None:
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
        locations = []
        for entity in entities:
            entity_type = self._classify_entity(entity)
            print(f"entity '{entity}' is of type {entity_type}")
            if entity_type == "location":
                locations.append(entity)
            #     self.memory.add_location_node(entity)
            #     print(self.memory._snapshot())
        if action_type in {"navigation", "start"}:
            self._update_location(observation, last_action, locations)
        print()
        print(self.memory._snapshot())
        print("=" * 20)


__all__ = ["MemoryManager"]
