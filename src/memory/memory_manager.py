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

    def update_memory(self, observation: str, last_action: str) -> None:
        print("=" * 20)
        print("Updating memory")
        print("Action type is", self._classify_action(last_action))
        entities = self._extract_relevant_entities(observation, last_action)
        new_locations: list[str] = []
        for entity in entities:
            print()
            print(f"updating memory for {entity=}")
            entity_type = self._classify_entity(entity)
            print(f"{entity_type=}")
            if entity_type == "location":
                self.memory.add_location_node(entity)
                new_locations.append(entity)
                print(self.memory._snapshot())
        if new_locations:
            self.memory.set_player_location(new_locations[-1])
        print("=" * 20)


__all__ = ["MemoryManager"]
