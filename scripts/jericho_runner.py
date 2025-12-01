#!/usr/bin/env python3
"""Run a Jericho game with a pluggable agent and memory backend."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make src/ available for imports when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent import BaseAgent, build_agent  # type: ignore
from llm import LLM, LlamaLLM  # type: ignore

try:
    from jericho import FrotzEnv
except ImportError as exc:  # pragma: no cover - environment dependency
    raise SystemExit(
        "Jericho is not installed. Please install with `pip install jericho` before running."
    ) from exc


def run_episode(
    env: FrotzEnv,
    agent: BaseAgent,
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
        agent.observe(step, action, observation, reward, info)

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
        "--agent", type=str, default="walkthrough", help="Agent type (walkthrough|llm)"
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
        "--torch-dtype",
        type=str,
        default="auto",
        help="Torch dtype for HF model loading (e.g., float16, bfloat16, auto)",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=32,
        help="Max new tokens for LLM action generation",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.5,
        help="Sampling temperature for LLM agent",
    )
    parser.add_argument(
        "--memory-mode",
        type=str,
        default="none",
        choices=["none", "llama"],
        help="Memory pipeline: none (disable) or llama (LLM extraction + grounding).",
    )
    parser.add_argument(
        "--extract-max-tokens",
        type=int,
        default=256,
        help="Max new tokens for extraction LLM completions.",
    )
    parser.add_argument(
        "--extract-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for extraction LLM (0 for deterministic).",
    )
    parser.add_argument(
        "--graphviz-dot",
        type=Path,
        default=None,
        help="Optional path to write GraphViz DOT export of the memory graph.",
    )
    parser.add_argument(
        "--graphviz-png",
        type=Path,
        default=None,
        help="Optional path to also render PNG (requires `dot` binary on PATH). If not provided, uses DOT path with .png.",
    )
    parser.add_argument(
        "--graphviz-include-inactive",
        action="store_true",
        help="Include inactive (closed) nodes/edges in GraphViz export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = FrotzEnv(str(args.game), seed=args.seed)
    shared_llm: Optional[LLM] = None
    if args.agent == "llm":
        shared_llm = LlamaLLM(
            model_id=args.model_id,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
        )

    agent = build_agent(args.agent, args=args, llm_client=shared_llm)

    trajectory = run_episode(
        env,
        agent,
        max_steps=args.max_steps,
        verbose=not args.quiet,
    )
    save_logs(args.text_log, args.json_log, trajectory)
    if args.graphviz_dot:
        agent.export_memory(
            args.graphviz_dot,
            include_inactive=args.graphviz_include_inactive,
            png_path=args.graphviz_png,
        )


if __name__ == "__main__":
    main()
