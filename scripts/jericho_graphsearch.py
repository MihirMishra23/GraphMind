#!/usr/bin/env python3
"""Small Jericho graph search that records action rewards and exports a DOT/PNG."""
from __future__ import annotations

import argparse
import hashlib
import logging
import pickle
import shutil
import subprocess
from collections import deque
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

# Make src/ available for imports when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from jericho import FrotzEnv
except ImportError as exc:  # pragma: no cover - environment dependency
    raise SystemExit(
        "Jericho is not installed. Please install with `pip install jericho` before running."
    ) from exc


Node = Dict[str, object]
Edge = Dict[str, object]


def _state_key(state_bytes: bytes) -> str:
    """
    Deterministic key for deduping visited env states.
    Handles bytes or arbitrary picklable tuples returned by Jericho.
    """
    try:
        if isinstance(state_bytes, (bytes, bytearray, memoryview)):
            raw = bytes(state_bytes)
        else:
            raw = pickle.dumps(state_bytes, protocol=4)
    except Exception:
        raw = repr(state_bytes).encode("utf-8", "replace")
    return hashlib.sha1(raw).hexdigest()


def _shorten(text: str, max_len: int = 120, max_lines: int = 2) -> str:
    """Keep node labels compact."""
    lines = text.strip().splitlines()
    lines = lines[:max_lines]
    trimmed = " / ".join(lines)
    if len(trimmed) > max_len:
        trimmed = trimmed[: max_len - 3] + "..."
    return trimmed


def _extract_direction(action: str) -> Optional[str]:
    """Best-effort direction parse from common navigation actions."""
    if not action:
        return None
    lower = action.strip().lower()
    if not lower:
        return None
    tokens = lower.split()
    if tokens[0] in {"go", "walk", "move"} and len(tokens) >= 2:
        return tokens[1]
    return tokens[0]


def _is_inverse_action(prev_action: Optional[str], candidate_action: str) -> bool:
    """Return True if candidate is the opposite direction of prev_action."""
    opposites = {
        "north": "south",
        "south": "north",
        "east": "west",
        "west": "east",
        "up": "down",
        "down": "up",
        "n": "s",
        "s": "n",
        "e": "w",
        "w": "e",
        "u": "d",
        "d": "u",
    }
    if not prev_action:
        return False
    prev_dir = _extract_direction(prev_action)
    cand_dir = _extract_direction(candidate_action)
    if not prev_dir or not cand_dir:
        return False
    return opposites.get(prev_dir) == cand_dir


def run_graph_search(
    env: FrotzEnv,
    max_depth: int,
    max_nodes: int,
    max_branch: int,
) -> Tuple[List[Node], List[Edge]]:
    """
    BFS over Jericho states, tracking rewards per action.

    Nodes: store observation snippet, cumulative reward, depth.
    Edges: store action label, immediate reward, and done flag.
    """
    observation, info = env.reset()
    root_state = env.get_state()
    nodes: List[Node] = [
        {
            "id": 0,
            "state": root_state,
            "obs": observation,
            "cum_reward": 0.0,
            "depth": 0,
            "done": False,
            "prev_action": None,
        }
    ]
    edges: List[Edge] = []
    visited = {_state_key(root_state): 0}
    queue = deque([0])

    while queue and len(nodes) < max_nodes:
        node_id = queue.popleft()
        node = nodes[node_id]
        depth = int(node["depth"])
        if depth >= max_depth:
            continue

        env.set_state(node["state"])  # type: ignore[arg-type]
        valid_actions = env.get_valid_actions()
        prev_action = node.get("prev_action")
        valid_actions = [
            a for a in valid_actions if not _is_inverse_action(prev_action, a)
        ]
        if max_branch > 0:
            valid_actions = valid_actions[:max_branch]

        for action in valid_actions:
            env.set_state(node["state"])  # type: ignore[arg-type]
            try:
                obs, reward, done, _info = env.step(action)
                if reward > 0:
                    print(f"REWARD YAYA\t{obs}")
            except Exception as exc:  # pragma: no cover - env robustness
                logging.debug(
                    "Action '%s' failed from node %s: %s", action, node_id, exc
                )
                continue

            child_state = env.get_state()
            key = _state_key(child_state)
            cum_reward = float(node["cum_reward"]) + float(reward)

            if key in visited:
                child_id = visited[key]
            else:
                child_id = len(nodes)
                visited[key] = child_id
                nodes.append(
                    {
                        "id": child_id,
                        "state": child_state,
                        "obs": obs,
                        "cum_reward": cum_reward,
                        "depth": depth + 1,
                        "done": bool(done),
                        "prev_action": action,
                    }
                )
                if not done:
                    # Prioritize exploring children reached via positive reward first.
                    if reward > 0:
                        queue.appendleft(child_id)
                    else:
                        queue.append(child_id)
                if len(nodes) >= max_nodes:
                    break

            edges.append(
                {
                    "src": node_id,
                    "dst": child_id,
                    "action": action,
                    "reward": float(reward),
                    "done": bool(done),
                }
            )
        else:
            # Only skip the outer break when max_nodes triggered.
            continue
        break

    return nodes, edges


def export_search_dot(
    nodes: List[Node],
    edges: List[Edge],
    dot_path: Path,
    png_path: Optional[Path] = None,
) -> None:
    """Minimal DOT exporter for the graph search traversal."""
    lines = ["digraph JerichoSearch {", "  rankdir=LR;"]

    for node in nodes:
        label_parts = [
            f"s{node['id']} (d{node['depth']})",
            f"R_total={node['cum_reward']:.2f}",
            _shorten(str(node.get("obs", ""))).replace('"', "'"),
        ]
        fill = "#c7c7c7" if not node.get("done") else "#f28e8c"
        lines.append(
            f'  "{node["id"]}" [shape=box, style=filled, fillcolor="{fill}", label="{"\\n".join(label_parts)}"];'
        )

    for edge in edges:
        reward = float(edge.get("reward", 0.0))
        action = str(edge.get("action", "")).replace('"', "'")
        done = edge.get("done")
        suffix = " (done)" if done else ""
        color = "green" if reward != 0.0 else "black"
        lines.append(
            f'  "{edge["src"]}" -> "{edge["dst"]}" [label="{action}\\n r={reward:.2f}{suffix}", color="{color}"];'
        )

    lines.append("}")
    dot_path = dot_path.resolve()
    dot_path.parent.mkdir(parents=True, exist_ok=True)
    dot_path.write_text("\n".join(lines))

    if png_path:
        dot_bin = shutil.which("dot")
        if not dot_bin:
            logging.warning("GraphViz `dot` binary not found; skipping PNG render.")
            return
        png_target = png_path if png_path.suffix else png_path.with_suffix(".png")
        png_target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [dot_bin, "-Tpng", str(dot_path), "-o", str(png_target)], check=False
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Graph search over Jericho states with reward-tracking."
    )
    parser.add_argument(
        "--game",
        type=Path,
        default=Path("data/jericho/z-machine-games-master/jericho-game-suite/zork1.z5"),
        help="Path to the .z machine file.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Optional random seed for Jericho env."
    )
    parser.add_argument(
        "--depth", type=int, default=15, help="Max search depth (root=0)."
    )
    parser.add_argument(
        "--branch", type=int, default=100, help="Max actions explored per node (0=all)."
    )
    parser.add_argument(
        "--max-nodes", type=int, default=200, help="Hard cap on explored nodes."
    )
    parser.add_argument(
        "--graphviz-dot",
        type=Path,
        default=Path("out/test.dot"),
        help="Path to write the DOT export.",
    )
    parser.add_argument(
        "--graphviz-png",
        type=Path,
        default=Path("out/test.png"),
        help="Optional PNG render path (requires GraphViz `dot`).",
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
    nodes, edges = run_graph_search(
        env, max_depth=args.depth, max_nodes=args.max_nodes, max_branch=args.branch
    )
    logging.info("Explored %d nodes and %d edges", len(nodes), len(edges))

    export_search_dot(nodes, edges, args.graphviz_dot, args.graphviz_png)
    logging.info("Wrote DOT to %s", args.graphviz_dot)
    if args.graphviz_png:
        png_path = (
            args.graphviz_png
            if args.graphviz_png.suffix
            else args.graphviz_png.with_suffix(".png")
        )
        if png_path.exists():
            logging.info("Wrote PNG to %s", png_path)


if __name__ == "__main__":
    main()
