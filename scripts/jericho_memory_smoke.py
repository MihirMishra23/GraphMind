#!/usr/bin/env python3
"""Smoke test runner that builds a WorldKG from Jericho walkthrough observations."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Make src/ available for imports when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent import WalkthroughAgent
from memory.entity_extractor import NaiveEntityRelationExtractor
from memory.kg_store import KGSnapshots
from memory.memory_manager import MemoryManager
from memory.schema import WorldKG
from memory.visualization import export_worldkg_dot

from jericho import FrotzEnv


def summarize_world(world: WorldKG) -> None:
    """Print a concise textual summary of the world KG."""
    graph = world.graph
    print("\n=== WorldKG Summary ===")
    print(f"Nodes: {len(graph.nodes)} | Edges: {len(graph.edges)}")
    print("\nNodes:")
    for node_id, data in graph.nodes(data=True):
        state = data.get("state") or {}
        state_txt = "; ".join(f"{k}={v}" for k, v in state.items()) if state else ""
        desc_parts = [f"[{data.get('type')}] {data.get('name', node_id)}"]
        if state_txt:
            desc_parts.append(f"state: {state_txt}")
        print(f"- {node_id}: " + " | ".join(desc_parts))

    print("\nEdges:")
    for src, dst, data in graph.edges(data=True):
        rel = data.get("type", "")
        attrs = {k: v for k, v in data.items() if k != "type"}
        attr_txt = f" {attrs}" if attrs else ""
        print(f"- {src} -[{rel}]-> {dst}{attr_txt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jericho memory smoke test (WorldKG)")
    parser.add_argument(
        "--game",
        type=Path,
        default=Path("data/jericho/z-machine-games-master/jericho-game-suite/zork1.z5"),
        help="Path to the .z machine file",
    )
    parser.add_argument(
        "--max-steps", type=int, default=50, help="Maximum steps to execute"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for Jericho env"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step prints (still summarizes graph at end)",
    )
    parser.add_argument(
        "--graphviz-dot",
        type=Path,
        default=None,
        help="Optional path to write GraphViz DOT export of the WorldKG.",
    )
    parser.add_argument(
        "--graphviz-png",
        type=Path,
        default=None,
        help="Optional path to render PNG via GraphViz `dot`. If omitted, uses DOT path with .png.",
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
    agent = WalkthroughAgent(env.get_walkthrough(), use_memory=False)

    world_kg = WorldKG()
    extractor = NaiveEntityRelationExtractor()
    memory_manager = MemoryManager(world_kg)
    snapshots = KGSnapshots()

    observation, info = env.reset()
    prev_obs: Optional[str] = None
    if not args.quiet:
        print("Initial observation:", observation)

    for step in range(args.max_steps):
        action = agent.act(observation, env.get_valid_actions())
        if action is None:
            break

        observation, reward, done, info = env.step(action)
        facts = extractor.extract(prev_obs, action, observation, world_kg, step)
        # Record observation node and link it in manager.
        obs_node_id = f"obs_{step}"
        world_kg.add_or_get_entity(
            canonical_id=obs_node_id,
            node_type="OBSERVATION",
            name=obs_node_id,
        )
        world_kg.graph.nodes[obs_node_id]["description"] = observation
        world_kg.graph.nodes[obs_node_id]["state"] = {"text": observation}
        world_kg.graph.nodes[obs_node_id]["last_updated_step"] = step

        memory_manager.decide_and_apply(facts, step=step, observation_node_id=obs_node_id)
        snapshots.store_snapshot(step, world_kg)
        prev_obs = observation

        if not args.quiet:
            print(f"\nStep {step}: action={action}")
            print(f"Observation: {observation}")
            print(f"Reward: {reward}, Done: {done}")
            print(f"Extracted {len(facts)} candidate facts")

        if done:
            break

    summarize_world(world_kg)
    if args.graphviz_dot:
        png_target = args.graphviz_png or args.graphviz_dot.with_suffix(".png")
        export_worldkg_dot(world_kg, args.graphviz_dot, png_target)


if __name__ == "__main__":
    main()
