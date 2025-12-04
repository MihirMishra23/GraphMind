from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from memory.schema import WorldKG, NodeType, EdgeType


def export_worldkg_dot(world: WorldKG, dot_path: Path, png_path: Optional[Path] = None) -> None:
    """
    Write a DOT representation of the WorldKG and optionally render to PNG via GraphViz `dot`.
    """
    lines = ["digraph WorldKG {", "  rankdir=LR;"]

    # Nodes
    for node_id, data in world.graph.nodes(data=True):
        ntype = data.get("type")
        label_parts = [f"{node_id}", f"[{ntype}] {data.get('name', node_id)}"]
        state = data.get("state") or {}
        # Special-case observation nodes to show actions.
        if ntype == NodeType.OBSERVATION.value:
            text = state.get("text") or data.get("description", "")
            actions = state.get("actions") or []
            if text:
                label_parts.append(text.replace("\"", "'"))
            if actions:
                label_parts.append("actions: " + ", ".join(actions))
            shape = "ellipse"
            color = "#6baed6"
        else:
            if state:
                state_txt = "\\n".join(f"{k}={v}" for k, v in state.items())
                label_parts.append(state_txt)
            shape = "box"
            color = "#c7c7c7"
        label = "\\n".join(label_parts)
        lines.append(f'  "{node_id}" [shape={shape}, style=filled, fillcolor="{color}", label="{label}"];')

    # Edges
    for src, dst, data in world.graph.edges(data=True):
        rel = data.get("type", "")
        edge_label = rel
        if rel == EdgeType.CONNECTED_TO.value and data.get("direction"):
            edge_label = f"{rel} ({data['direction']})"
        if rel == EdgeType.ACTION.value:
            edge_label = data.get("name") or data.get("command") or rel
        lines.append(f'  "{src}" -> "{dst}" [label="{edge_label}"];')

    lines.append("}")
    dot_path = dot_path.resolve()
    dot_path.parent.mkdir(parents=True, exist_ok=True)
    dot_path.write_text("\n".join(lines))

    # Optional PNG render using GraphViz if available.
    if png_path:
        dot_bin = shutil.which("dot")
        if not dot_bin:
            logging.warning("GraphViz `dot` binary not found; skipping PNG render.")
            return
        png_target = png_path if png_path.suffix else png_path.with_suffix(".png")
        Path(png_target).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([dot_bin, "-Tpng", str(dot_path), "-o", str(png_target)], check=False)


__all__ = ["export_worldkg_dot"]
