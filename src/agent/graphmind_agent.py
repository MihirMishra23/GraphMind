from __future__ import annotations

"""Agent that queries an LLM to decide the next action."""
from typing import List, Optional, Sequence

from llm import LLM

from .base import BaseAgent
from memory.memory_manager import MemoryManager


DEFAULT_SYSTEM_PROMPT = """You are playing a classic text-based interactive fiction game. Your goal is to explore, solve puzzles, collect treasures, and win. You are the PLAYER.

Each turn you receive:
- Recent history (last 8 turns) of actions and observations, including the latest observation resulting from your last action.
- The current known state of the world gathered from prior interactions

Your task:
- Think step by step about the best next action using the latest observation, last action, and recent history.
- If unsure, prefer exploring new states (e.g., new directions, inspecting new objects, or using “look”/“inventory”) rather than repeating loops.
- Avoid random or purposeless moves; favor progress toward exploration and puzzles.
- After reasoning, output exactly ONE concise game command (1-3 words).

Output format:
- Include your reasoning. Keep your reasoning within a few sentences.
- End with the final command wrapped exactly as:
  <start> your command <end>"""

# EXAMPLES:

# Observation: You are in a dark room.
# Valid Actions: look, wait
# Reasoning: Looking lets us explore.
# <start> look <end>

# Observation: A mailbox is here.
# Valid Actions: open mailbox, do nothing, leave
# Reasoning: I should check the mail.
# <start> open mailbox <end>"""


class GraphMindAgent(BaseAgent):
    def __init__(
        self,
        llm: LLM,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        history_horizon: int = 8,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.system_prompt = system_prompt
        self.history_horizon = history_horizon
        self._last_action: str = "start"
        self.memory_manager = MemoryManager(llm)
        self._state_actions: dict[str, set[str]] = {}

    def reset(self, env: object) -> None:
        super().reset(env)
        self._last_action = "start"

    def observe(self, observation: str):
        self.memory_manager.update_memory(observation, self._last_action)
        if self._last_action:
             self._recent_steps.append({
                 "action": self._last_action,
                 "observation": observation,
             })

    def act(
        self,
        observation: str,
        action_candidates: List[str],
        override: Optional[str] = None,
    ) -> Optional[str]:
        if override:
            self._last_action = override
            return override
        recent_lines = self._get_recent_history_lines(self.history_horizon)

        current_state = str(hash(self.memory_manager.memory))
        tried_actions = self._state_actions.get(current_state, set())
        unseen_candidates = [
            cand for cand in action_candidates if cand.lower() not in tried_actions
        ]

        action = self.propose_action(
            observation,
            recent_lines,
            action_candidates=unseen_candidates or action_candidates,
        )
        # action = self._avoid_recent_repeat(action, action_candidates, observation)

        self._last_action = action
        return action or "look"

    def propose_action(
        self,
        obs: str,
        recent_history: list[str],
        action_candidates: list[str],
    ) -> str:
        history_block = "\n".join(recent_history[-self.history_horizon*2 :])
        snapshot = self.memory_manager.memory.to_dict()
        player = snapshot.get("player", {})
        location = player.get("location")
        inventory = player.get("inventory", [])
        locations = list(snapshot.get("locations", {}).keys())
        objects = snapshot.get("objects", {})
        entities_context = self.memory_manager.get_recent_entities_context()
        candidates_str = ""
        if action_candidates:
             candidates_str = "\nCurrent Valid Actions: " + ", ".join(action_candidates)
        prompt = (
            f"{self.system_prompt}\n\n"
            "Use the current world state and history to decide the next action.\n"
            # "Here is the recent history:\n"
            f"Here is the recent history:\n{history_block}\n\n"
            f"The last action-observation pair is your latest action taken and the current/latest observation resulting from that action.\n"
            "Here is the current known world state roughly gathered from prior interactions:\n"
            f"Player location: {location}\n"
            f"Inventory: {inventory}\n"
            f"Known locations: {locations}\n"
            f"Known objects and states: {objects}\n"
            f"Latest observation reminder:\n{obs}\n"
            f"{candidates_str}\n\n"
            "Think step by step, then output one action that is exactly letter for letter the same as one of the valid actions (nothing more, nothing less).\n"
            "Format:\n"
            "Reasoning: your reasoning\n"
            "<start> your action <end>\n\n"
        )

        print(f"\n[DEBUG] PROMPT SENT TO LLM:\n...{prompt}\n")

        completion = self.llm.generate(
            prompt,
            max_tokens=256,
            stop=["<end>"],
        )
        text = completion.strip()

        print(f"[DEBUG] RAW LLM OUTPUT:\n{completion}\n")

        lower_text = text.lower()
        start_idx = lower_text.find("<start>")
        if start_idx != -1:
            content = text[start_idx + len("<start>") :]
            end_idx = content.lower().find("<end>")
            if end_idx != -1:
                content = content[:end_idx]
            
            cleaned = content.strip()
        else:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            cleaned = lines[0] if lines else ""

        cleaned_lower = cleaned.lower()
        print()
        print(f"{cleaned_lower=}")
        print()
        return cleaned_lower
        # Exact match
        for cand in action_candidates:
            if cleaned_lower == cand.lower():
                return cand
        # Prefix/substring match
        for cand in action_candidates:
            if cleaned_lower.startswith(cand.lower()) or cand.lower().startswith(
                cleaned_lower
            ):
                return cand
        # Fallback to first candidate
        return action_candidates[0]

    def _avoid_recent_repeat(
        self,
        action: str,
        action_candidates: Sequence[str],
        observation: Optional[str] = None,
    ) -> str:
        """
        Heuristic to break out of short loops: if the model keeps repeating the
        same action in the last few turns, pick the next available candidate.
        """
        if not action_candidates:
            return action
        recent_steps = self._recent_steps[-3:]
        recent_actions = [step["action"] for step in recent_steps]
        action_lower = action.lower()

        # If we have repeated the same action recently, pick a different candidate.
        if recent_actions and all(a.lower() == action_lower for a in recent_actions):
            for cand in action_candidates:
                if cand.lower() != action_lower:
                    return cand

        # Break simple open/close toggle loops on the same object.
        if len(recent_actions) >= 2:
            last = recent_actions[-1].lower()
            prev = recent_actions[-2].lower()
            verb_last, obj_last = self._split_verb_object(last)
            verb_prev, obj_prev = self._split_verb_object(prev)
            verb_curr, obj_curr = self._split_verb_object(action_lower)
            toggles = {
                ("open", "close"),
                ("close", "open"),
                ("turn on", "turn off"),
                ("turn off", "turn on"),
            }
            if (
                (verb_prev, verb_last) in toggles
                and obj_prev == obj_last
                and obj_curr == obj_last
            ):
                for cand in action_candidates:
                    cand_lower = cand.lower()
                    cand_verb, cand_obj = self._split_verb_object(cand_lower)
                    if cand_obj != obj_last or cand_verb not in {verb_prev, verb_last}:
                        return cand

        # If the latest observation did not change from the prior step and we are about
        # to repeat the same action, choose an alternative to escape loops.
        if observation:
            obs_clean = observation.strip().lower()
            if recent_steps:
                last_obs = recent_steps[-1]["observation"].strip().lower()
                if (
                    obs_clean == last_obs
                    and recent_actions
                    and recent_actions[-1].lower() == action_lower
                ):
                    for cand in action_candidates:
                        if cand.lower() not in {a.lower() for a in recent_actions}:
                            return cand
                        if cand.lower() != action_lower:
                            return cand
        return action

    def _split_verb_object(self, action: str) -> tuple[str, str]:
        """
        Naively split an action into (verb, object) to detect toggles.
        """
        parts = action.split()
        if not parts:
            return "", ""
        if len(parts) == 1:
            return parts[0], ""
        return parts[0], " ".join(parts[1:])
