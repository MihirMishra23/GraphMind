#!/usr/bin/env python3
"""Run a Jericho game with a pluggable agent."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make src/ available for imports when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent import BaseAgent, build_agent  # type: ignore

try:
    from jericho import FrotzEnv
except ImportError as exc:  # pragma: no cover - environment dependency
    raise SystemExit(
        "Jericho is not installed. Please install with `pip install jericho` before running."
    ) from exc

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_episode(
    env: FrotzEnv,
    agent: BaseAgent,
    max_steps: int,
    verbose: bool = True,
    manual: bool = False,
) -> List[Dict[str, Any]]:
    observation, info = env.reset()
    agent.reset(env)
    if verbose:
        print(f"Step 0: action: start")
        print((observation, info))

    trajectory: List[Dict[str, Any]] = []

    agent.observe(observation)

    for step in range(1, max_steps):
        valid_actions = env.get_valid_actions()
        if manual:
            print("\n--- Manual step ---")
            print(f"Observation:\n{observation}")
            print("Valid actions:")
            for a in valid_actions:
                print(f"- {a}")
            try:
                action = input("Enter action: ").strip()
            except EOFError:
                action = ""
            if not action:
                break
        else:
            action = agent.act(observation, valid_actions)
        if action is None:
            break

        observation, reward, done, info = env.step(action)
        agent.observe(observation)

        record = {
            "step": step,
            "action": action,
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info,
        }
        trajectory.append(record)

        if verbose:
            print(f"Step {step}: action {action}")
            print((observation, info))

        if done:
            break

    return trajectory


def save_logs(
    text_log: Optional[Path], json_log: Optional[Path], trajectory: List[Dict[str, Any]]
) -> None:
    if text_log:
        lines = []
        for row in trajectory:
            lines.append(f"Step {row['step']}: action {row['action']}")
            lines.append(
                str((row["observation"], row["reward"], row["done"], row["info"]))
            )
        text_log.write_text("\n".join(lines))

    if json_log:
        json_log.write_text(json.dumps(trajectory, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jericho runner")
    parser.add_argument(
        "--game",
        type=Path,
        default=Path("data/jericho/z-machine-games-master/jericho-game-suite/zork1.z5"),
        help="Path to the .z machine file",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="walkthrough",
        choices=["walkthrough", "llm", "graphmind"],
        help="Agent type (walkthrough|llm|graphmind)",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Bypass agent decisions and manually input actions each step.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=400, help="Maximum steps to execute"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for Jericho env"
    )
    parser.add_argument(
        "--text-log",
        type=Path,
        default=None,
        help="Optional plain-text trajectory log path",
    )
    parser.add_argument(
        "--json-log", type=Path, default=None, help="Optional JSON trajectory log path"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout step printing (still logs if paths provided)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Hugging Face model id for the LLM agent",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map for HF model loading (auto|cuda|mps|cpu; auto prefers cuda>mps>cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Torch dtype for HF model loading (e.g., float16, bfloat16, auto)",
    )
    parser.add_argument(
        "--llm-backend",
        type=str,
        default="llama",
        choices=["llama", "openai"],
        help="Backend LLM provider: local Llama HF model or OpenAI chat model.",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI chat model to use when --llm-backend=openai.",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="Optional OpenAI API key (otherwise uses OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    env = FrotzEnv(str(args.game), seed=args.seed)
    shared_llm = None
    if args.agent == "walkthrough":
        agent = build_agent(
            args.agent,
            args=args,
            llm_client=shared_llm,
            walkthrough=env.get_walkthrough(),
        )
    else:
        # For llm-based agents, build_agent will create the LLM client if missing.
        agent = build_agent(args.agent, args=args, llm_client=shared_llm)

    trajectory = run_episode(
        env,
        agent,
        max_steps=args.max_steps,
        verbose=not args.quiet,
        manual=args.manual,
    )
    save_logs(args.text_log, args.json_log, trajectory)


if __name__ == "__main__":
    main()
