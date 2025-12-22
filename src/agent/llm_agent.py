from __future__ import annotations

"""Agent that queries an LLM to decide the next action."""
from typing import List, Optional, Sequence

from llm import LLM

from .base import BaseAgent


# DEFAULT_SYSTEM_PROMPT = """You are playing a text adventure game. You are the PLAYER.
# Your goal is to win the game.

# RULES:
# 1. Do NOT act as a narrator or guide.
# 2. Do NOT output "Example:" or "Note:".
# 3. Output your reasoning, then the final command wrapped in <start> and <end>.

# EXAMPLES:

# Observation: You are in a dark room.
# <start> look <end>

# Observation: A mailbox is here.
# Reasoning: I should check the mail.
# <start> open mailbox <end>

# Your turn:"""

DEFAULT_SYSTEM_PROMPT = """You are playing a classic text-based interactive fiction game. Your goal is to explore, solve puzzles, collect treasures, and win. You are the PLAYER.

Each turn you receive:
- Recent history (last 8 turns) of actions and observations, including the latest observation resulting from your last action.

Your task:
- Think step by step about the best next action using the latest observation, last action, and recent history.
- If unsure, prefer exploring new states (e.g., new directions, inspecting new objects, or using “look”/“inventory”) rather than repeating loops.
- Avoid random or purposeless moves; favor progress toward exploration and puzzles.
- After reasoning, output exactly ONE concise game command (1-3 words).

Output format:
- Include your reasoning. Keep your reasoning within a few sentences.
- Write with the final command wrapped exactly as:
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


class LLMAgent(BaseAgent):
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

    def reset(self, env: object) -> None:
        super().reset(env)
        self._last_action = "start"

    def observe(self, observation: str) -> None:
        # If we have performed an action previously, record the result
        if self._last_action is not None:
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
        action = self.propose_action(
            observation,
            recent_lines,
            action_candidates=action_candidates,
        )
        action = self._avoid_recent_repeat(action, action_candidates, observation)
        self._last_action = action
        return action or "look"

    def propose_action(
        self,
        obs: str,
        recent_history: list[str],
        action_candidates: Optional[Sequence[str]] = None,
    ) -> str:
        history_block = "\n".join(recent_history[-self.history_horizon*2 :])
        # assert self.history_horizon == 8
        candidates_str = ""
        if action_candidates:
             candidates_str = "\nCurrent Valid Actions: " + ", ".join(action_candidates)
        prompt = (
            f"{self.system_prompt}\n\n"
            "Use recent history and the latest observation to decide the next action.\n"
            f"Recent history:\n{history_block}\n\n"
            f"The last action-observation pair is your latest action taken and the current/latest observation resulting from that action.\n"
            # f"Latest observation:\n{obs}\n"
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
        completion = completion.strip()

        print(f"[DEBUG] RAW LLM OUTPUT:\n{completion}\n")

        print(f"{completion=}")
        if not action_candidates:
            return completion

        # Extract between <start> ... <end> if present; otherwise first non-empty line.
        lower_text = completion.lower()
        start_idx = lower_text.find("<start>")
        if start_idx != -1:
            # cleaned = completion[start_idx + len("<start>") :].strip()
            # 1. Get the text strictly AFTER <start>
            content = completion[start_idx + len("<start>") :]
            
            # 2. Find <end> within that content and slice it off
            end_idx = content.lower().find("<end>")
            if end_idx != -1:
                content = content[:end_idx]
            
            cleaned = content.strip()
        else:
            lines = [ln.strip() for ln in completion.splitlines() if ln.strip()]
            cleaned = lines[0] if lines else ""

        cleaned_lower = cleaned.lower()
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
