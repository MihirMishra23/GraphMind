from __future__ import annotations

"""Lightweight memory manager for extracting entities from observations."""

import json
from typing import Optional

from llm import LLM


class MemoryManager:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def extract_relevant_entities(
        self, observation: str, last_action: Optional[str] = None
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

        return entities


__all__ = ["MemoryManager"]
