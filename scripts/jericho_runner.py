#!/usr/bin/env python3
"""Run a Jericho game with a pluggable agent and memory backend."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Make src/ available for imports when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent import BaseAgent, LLMAgent, WalkthroughAgent  # type: ignore
from memory import NullMemory  # type: ignore
from llm import LlamaLLM  # type: ignore

try:
    from jericho import FrotzEnv
except ImportError as exc:  # pragma: no cover - environment dependency
    raise SystemExit(
        "Jericho is not installed. Please install with `pip install jericho` before running."
    ) from exc


def build_agent(name: str, memory: Optional[object], args: argparse.Namespace) -> BaseAgent:
    if name == "walkthrough":
        return WalkthroughAgent(memory=memory)
    if name == "llm":
        device_map = resolve_device_map(args.device_map)
        llm_client = LlamaLLM(
            model_id=args.model_id,
            device_map=device_map,
            torch_dtype=args.torch_dtype,
        )
        return LLMAgent(
            llm=llm_client,
            memory=memory,
            max_tokens=args.llm_max_tokens,
            temperature=args.llm_temperature,
        )
    raise ValueError(f"Unsupported agent type: {name}")


def resolve_device_map(requested: str) -> str:
    """Pick device map with priority: cuda > mps > cpu, unless explicitly set."""
    req = requested.lower()
    if req != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_episode(
    env: FrotzEnv,
    agent: BaseAgent,
    memory: NullMemory,
    max_steps: int,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    observation, info = env.reset()
    agent.reset(env)
    trajectory: List[Dict[str, Any]] = []

    for step in range(max_steps):
        action = agent.act(observation, info)
        if action is None:
            break

        observation, reward, done, info = env.step(action)
        memory.observe(step, action, observation, reward, info)

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
            print((observation, reward, done, info))

        if done:
            break

    return trajectory


def save_logs(text_log: Optional[Path], json_log: Optional[Path], trajectory: List[Dict[str, Any]]) -> None:
    if text_log:
        lines = []
        for row in trajectory:
            lines.append(f"Step {row['step']}: action {row['action']}")
            lines.append(str((row["observation"], row["reward"], row["done"], row["info"])))
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
    parser.add_argument("--agent", type=str, default="walkthrough", help="Agent type (walkthrough|llm)")
    parser.add_argument("--max-steps", type=int, default=400, help="Maximum steps to execute")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for Jericho env")
    parser.add_argument("--text-log", type=Path, default=None, help="Optional plain-text trajectory log path")
    parser.add_argument("--json-log", type=Path, default=None, help="Optional JSON trajectory log path")
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress stdout step printing (still logs if paths provided)"
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
        "--torch-dtype",
        type=str,
        default="auto",
        help="Torch dtype for HF model loading (e.g., float16, bfloat16, auto)",
    )
    parser.add_argument("--llm-max-tokens", type=int, default=32, help="Max new tokens for LLM action generation")
    parser.add_argument("--llm-temperature", type=float, default=0.5, help="Sampling temperature for LLM agent")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = FrotzEnv(str(args.game), seed=args.seed)
    memory = NullMemory()
    agent = build_agent(args.agent, memory=memory, args=args)

    trajectory = run_episode(env, agent, memory, max_steps=args.max_steps, verbose=not args.quiet)
    save_logs(args.text_log, args.json_log, trajectory)


if __name__ == "__main__":
    main()
