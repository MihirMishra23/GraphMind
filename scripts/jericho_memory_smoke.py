#!/usr/bin/env python3
"""Smoke test runner that builds a graph from Jericho walkthrough observations."""
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

from agent import WalkthroughAgent  # type: ignore
from memory import (
    ExtractedEntity,
    ExtractedEvent,
    ExtractedRelation,
    ExtractionResult,
    GraphStore,
    Grounder,
    parse_extraction_output,
)  # type: ignore
from tools import export_graphviz  # type: ignore

try:
    from jericho import FrotzEnv
except ImportError as exc:  # pragma: no cover - environment dependency
    raise SystemExit(
        "Jericho is not installed. Please install with `pip install jericho` before running."
    ) from exc


def naive_extract(
    observation: str, history: List[str], turn_id: int
) -> ExtractionResult:
    """Very lightweight heuristic extractor to avoid loading a real LLM."""
    entities: List[ExtractedEntity] = []
    relations: List[ExtractedRelation] = []
    events: List[ExtractedEvent] = []

    seen_entities = set()
    lowered = observation.lower()
    location_name: Optional[str] = None

    def add_entity(name: str, ent_type: str, confidence: float) -> None:
        if not name or name in seen_entities:
            return
        entities.append(
            ExtractedEntity(
                name=name,
                type=ent_type,
                aliases=[name],
                confidence=confidence,
                time=turn_id,
            )
        )
        seen_entities.add(name)

    # Guess a location from "you are in/at..." patterns.
    for marker in ["you are in ", "you are at "]:
        if marker in lowered:
            idx = lowered.index(marker) + len(marker)
            fragment = observation[idx:].split(".")[0].strip()
            if fragment:
                add_entity(fragment, "location", 0.55)
                location_name = fragment
            break

    # Grab noun-ish chunks following articles.
    for token in ["a ", "an ", "the "]:
        start = 0
        while True:
            idx = lowered.find(token, start)
            if idx == -1:
                break
            chunk = observation[idx + len(token) :].split(",")[0].split(".")[0].strip()
            chunk = " ".join(chunk.split()[:3])  # keep first few words
            add_entity(chunk, "object", 0.45)
            start = idx + len(token)

    # Connect objects to location if present.
    if location_name:
        for ent in entities:
            if ent.name == location_name:
                continue
            relations.append(
                ExtractedRelation(
                    source=location_name,
                    target=ent.name,
                    rel_label="contains",
                    confidence=0.4,
                    time=turn_id,
                )
            )

    # Record the observation text as an event description.
    events.append(
        ExtractedEvent(
            description=observation[:180],
            participants=[e.name for e in entities],
            properties={},
            confidence=0.4,
            time=turn_id,
        )
    )

    return ExtractionResult(entities=entities, relations=relations, events=events)


def summarize_graph(store: GraphStore) -> None:
    print("\n=== Graph Summary ===")
    print(f"Nodes: {len(store.nodes)} | Edges: {len(store.edges)}")
    print("\nNodes:")
    for node in store.nodes.values():
        status = "active" if node.valid_to is None else f"closed@{node.valid_to}"
        print(
            f"- {node.node_id} [{node.node_type}] aliases={node.aliases} props={node.properties} ({status})"
        )
    print("\nEdges:")
    for edge in store.edges.values():
        status = "active" if edge.valid_to is None else f"closed@{edge.valid_to}"
        print(
            f"- {edge.edge_id} {edge.source} -[{edge.rel_label}]-> {edge.target} ({status})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jericho memory smoke test")
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
        "--use-llm-output",
        type=str,
        default=None,
        help="Optional path to a file containing an LLM extraction completion to parse instead of heuristic.",
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
        help="Optional path to also render PNG (requires `dot` binary). If not provided, uses DOT path with .png.",
    )
    parser.add_argument(
        "--graphviz-include-inactive",
        action="store_true",
        help="Include inactive (closed) nodes/edges in GraphViz export.",
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
    agent = WalkthroughAgent(env.get_walkthrough())
    store = GraphStore()
    grounder = Grounder(store)

    observation, info = env.reset()
    history: List[str] = []
    if not args.quiet:
        print("Initial observation:", observation)

    for step in range(args.max_steps):
        action = agent.act(observation, env.get_valid_actions())
        if action is None:
            break

        observation, reward, done, info = env.step(action)
        history.append(f"{action} -> {observation}")

        # Either parse provided LLM output or fallback to the heuristic extractor.
        if args.use_llm_output:
            text = Path(args.use_llm_output).read_text()
            extraction = parse_extraction_output(text)
        else:
            extraction = naive_extract(observation, history, turn_id=step)

        grounder.ground(extraction, turn_id=step, action=action)

        if not args.quiet:
            print(f"\nStep {step}: action={action}")
            print(f"Observation: {observation}")
            print(f"Reward: {reward}, Done: {done}")

        if done:
            break

    summarize_graph(store)
    if args.graphviz_dot:
        export_graphviz(
            store,
            args.graphviz_dot,
            include_inactive=args.graphviz_include_inactive,
            render_png=bool(args.graphviz_png),
            png_path=args.graphviz_png,
        )


if __name__ == "__main__":
    main()
