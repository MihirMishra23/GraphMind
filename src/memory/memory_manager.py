from __future__ import annotations

"""Lightweight memory manager for extracting entities from observations."""

import json
from typing import Optional, Literal

from llm import LLM
from memory import Memory


class MemoryManager:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self.memory = Memory()

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

    def _classify_entity(self, entity) -> Literal["location", "object"]:
        prompt = (
            "I want to know the type of the entity. Respond with the correct type of the entity\n"
            "Respond with exactly one word: 'location' or 'object'.\n\n"
            f"Between location and object, {entity} is of type "
        )
        completion = self.llm.generate(prompt, max_tokens=20, stop=None).strip().lower()
        if "location" in completion:
            return "location"
        if "object" in completion:
            return "object"
        raise Exception(
            f"Error when classifying entity {entity} - received {completion=}"
        )

    def _classify_action(
        self, action
    ) -> Literal["navigation", "manipulation", "perception", "start"]:
        if action == "start":
            return "start"
        prompt = (
            "I want to know the type of the action. Respond with the correct type of the action.\n"
            "There are 3 types: navigation (movement like go north, enter), manipulation (interacting with objects like open door, take key), perception (observing like look, examine, inventory).\n"
            "Respond with exactly one of: navigation, manipulation, perception. No other text.\n\n"
            "Example 1:\n"
            "Action 'go north' is of type 'navigation'<END>\n\n"
            "Example 2:\n"
            "Action 'open the mailbox' is of type 'manipulation'<END>\n\n"
            "Example 3:\n"
            "Action 'look around' is of type 'perception'<END>\n\n"
            f"Action '{action}' is of type "
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

    def _update_location(
        self, observation: str, navigation_action: str, potential_locations: list
    ) -> None:
        """
        Use the LLM to infer the current location from the observation, update the
        player's location, and add a directional edge from the previous location.
        """
        prompt = (
            "You are playing a text adventure game."
            "Identify your current location from this observation.\n"
            "Only respond with the location name (short word or phrase, normally capitalized), and then the <END> token.\n"
            f"Observation: {observation}\n"
            "Based on the given observation, my location is: "
        )
        print(f"{prompt=}")
        completion = self.llm.generate(prompt, max_tokens=16, stop=["<END>"]).strip()
        print(f"{completion=}")
        if not completion:
            raise Exception("Unable to extract location from observation")
        prev_loc = str(self.memory.player.get("location") or "")
        self.memory.add_location_node(completion)
        self.memory.set_player_location(completion)
        print(f"{prev_loc=}")
        print(self.memory._snapshot())
        self.memory.add_location_edge(prev_loc, navigation_action, completion)

    def update_memory(self, observation: str, last_action: str) -> None:
        print("=" * 20)
        print("Updating memory")
        action_type = self._classify_action(last_action)
        entities = self._extract_relevant_entities(observation, last_action)
        locations = []
        for entity in entities:
            # print()
            # print(f"updating memory for {entity=}")
            entity_type = self._classify_entity(entity)
            # print(f"{entity_type=}")
            if entity_type == "location":
                locations.append(entity)
            #     self.memory.add_location_node(entity)
            #     print(self.memory._snapshot())
        if action_type == "navigation" or "start":
            self._update_location(observation, last_action, locations)
        print()
        print(self.memory._snapshot())
        print("=" * 20)


__all__ = ["MemoryManager"]
