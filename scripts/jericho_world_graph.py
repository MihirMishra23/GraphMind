#!/usr/bin/env python3
"""Build a graph from `FrotzEnv.get_world_objects()` and visualize it."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# Make src/ available for imports when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from memory import GraphStore  # type: ignore
from tools import export_graphviz  # type: ignore

try:
    from jericho import FrotzEnv
except ImportError as exc:  # pragma: no cover - environment dependency
    raise SystemExit(
        "Jericho is not installed. Please install with `pip install jericho` before running."
    ) from exc


Scalar = (str, int, float, bool)


def _get_field(obj: Any, *candidates: str) -> Any:
    """Return the first present attribute/key for any candidate name."""
    if isinstance(obj, dict):
        for name in candidates:
            if name in obj and obj[name] is not None:
                return obj[name]
    for name in candidates:
        if hasattr(obj, name):
            val = getattr(obj, name)
            if not callable(val):
                return val
    return None


def _iter_world_objects(world: Any) -> List[Tuple[Any, Any]]:
    """Normalize the world object container into a list of (key, obj)."""
    if isinstance(world, dict):
        return list(world.items())
    if isinstance(world, list):
        return list(enumerate(world))
    raise TypeError(f"Unsupported world object container type: {type(world)}")


def _child_ids(raw: Any) -> Iterable[str]:
    """Extract child identifiers from common Jericho fields."""
    children = _get_field(raw, "children", "inventory", "contents", "objs")
    if children is None:
        return []
    if isinstance(children, dict):
        return [str(k) for k in children.keys()]
    if isinstance(children, (list, tuple, set)):
        return [str(c) for c in children]
    return [str(children)]


def _scalar_properties(raw: Any) -> Dict[str, Any]:
    """Keep only lightweight scalar properties for graph node labels."""
    props: Dict[str, Any] = {}
    data: Dict[str, Any] = {}
    if isinstance(raw, dict):
        data = raw
    else:
        for attr in dir(raw):
            if attr.startswith("_"):
                continue
            try:
                val = getattr(raw, attr)
            except Exception:
                continue
            if callable(val):
                continue
            data[attr] = val
    for key, val in data.items():
        if key in {"children", "inventory", "contents", "parent", "location", "objs"}:
            continue
        if isinstance(val, Scalar) or val is None:
            props[key] = val
        elif isinstance(val, (list, tuple, set)) and val and all(
            isinstance(x, Scalar) or x is None for x in val
        ):
            props[key] = list(val)[:6]
    return props


def build_graph_from_world(world: Any) -> GraphStore:
    """Convert the output of `get_world_objects` into a GraphStore."""
    store = GraphStore()
    items = _iter_world_objects(world)

    # First pass: create nodes for every known object.
    id_to_node: Dict[str, str] = {}
    for key, raw in items:
        obj_id = str(_get_field(raw, "id", "object_id", "obj_id") or key)
        name = _get_field(raw, "name", "object", "object_name", "shortname", "vocab")
        aliases = [str(name)] if name else [obj_id]
        props = _scalar_properties(raw)
        node = store.add_node(node_type="world_object", aliases=aliases, properties=props)
        id_to_node[obj_id] = node.node_id

    # Second pass: add containment edges based on parent/child hints.
    seen_edges = set()
    for key, raw in items:
        obj_id = str(_get_field(raw, "id", "object_id", "obj_id") or key)
        node_id = id_to_node[obj_id]

        parent = _get_field(raw, "parent", "location", "loc", "owner")
        parent_id = None
        if parent is not None:
            parent_id = str(_get_field(parent, "id", "object_id", "obj_id") or parent)

        if parent_id and parent_id in id_to_node:
            edge_key = (parent_id, obj_id)
            if edge_key not in seen_edges:
                store.add_edge(
                    source=id_to_node[parent_id],
                    target=node_id,
                    rel_label="contains",
                    provenance={"source": "world_model"},
                )
                seen_edges.add(edge_key)

        for child_id in _child_ids(raw):
            if child_id not in id_to_node:
                continue
            edge_key = (obj_id, child_id)
            if edge_key in seen_edges:
                continue
            store.add_edge(
                source=node_id,
                target=id_to_node[child_id],
                rel_label="contains",
                provenance={"source": "world_model"},
            )
            seen_edges.add(edge_key)

    return store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Jericho world objects as a graph.")
    parser.add_argument(
        "--game",
        type=Path,
        default=Path("data/jericho/z-machine-games-master/jericho-game-suite/zork1.z5"),
        help="Path to the .z machine file (defaults to the smoke-test game).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for Jericho env.")
    parser.add_argument(
        "--graphviz-dot",
        type=Path,
        default=Path("out/world_objects.dot"),
        help="Path to write the DOT export (set to empty to skip writing).",
    )
    parser.add_argument(
        "--graphviz-png",
        type=Path,
        default=None,
        help="Optional path to render a PNG (requires `dot`). Defaults to DOT path with .png.",
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
    env.reset()
    world = env.get_world_objects()
    store = build_graph_from_world(world)

    print(f"Graph has {len(store.nodes)} nodes and {len(store.edges)} edges")
    if args.graphviz_dot:
        png_path = args.graphviz_png or args.graphviz_dot.with_suffix(".png")
        export_graphviz(
            store,
            args.graphviz_dot,
            include_inactive=args.graphviz_include_inactive,
            render_png=True,
            png_path=png_path,
        )
        print(f"Wrote DOT to {args.graphviz_dot}")
        if Path(png_path).exists():
            print(f"Wrote PNG to {png_path}")


if __name__ == "__main__":
    main()
