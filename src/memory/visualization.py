from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from memory.schema import WorldKG


def export_worldkg_dot(world: WorldKG, dot_path: Path, png_path: Optional[Path] = None) -> None:
    """
    Write a DOT representation of the WorldKG and optionally render to PNG via GraphViz `dot`.
    """
    lines = ["digraph WorldKG {", "  rankdir=LR;"]

    # Nodes
    for node_id, data in world.graph.nodes(data=True):
        label_parts = [f"{node_id}", f"[{data.get('type')}] {data.get('name', node_id)}"]
        state = data.get("state") or {}
        if state:
            state_txt = "\\n".join(f"{k}={v}" for k, v in state.items())
            label_parts.append(state_txt)
        label = "\\n".join(label_parts)
        lines.append(f'  "{node_id}" [shape=box, label="{label}"];')

    # Edges
    for src, dst, data in world.graph.edges(data=True):
        rel = data.get("type", "")
        edge_label = rel
        if rel == "CONNECTED_TO" and data.get("direction"):
            edge_label = f"{rel} ({data['direction']})"
        if rel == "ACTION":
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
