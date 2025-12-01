"""GraphViz export helpers for GraphStore."""
from __future__ import annotations

import subprocess
import shutil
from pathlib import Path
from typing import Optional

from memory import GraphStore


def export_graphviz(
    store: GraphStore,
    dot_path: Path,
    include_inactive: bool = False,
    render_png: bool = False,
    png_path: Optional[Path] = None,
) -> Path:
    """Write the graph to a DOT file and optionally render to PNG if `dot` is available.

    Args:
        store: GraphStore instance to export.
        dot_path: Path to write the DOT file.
        include_inactive: If False, only export active nodes/edges.
        render_png: If True and GraphViz `dot` is on PATH, also emit a PNG.
        png_path: Optional override for the PNG output path.

    Returns:
        Path to the written DOT file.
    """
    dot_lines = ["digraph memory {", '  rankdir="LR";', '  node [shape=box, style=rounded];']

    nodes = store.nodes.values() if include_inactive else store.active_nodes()
    edges = store.edges.values() if include_inactive else store.active_edges()

    for node in nodes:
        label_alias = node.aliases[0] if node.aliases else node.node_id
        status = "active" if node.valid_to is None else f"closed@{node.valid_to}"
        label = f"{label_alias}\\n[{node.node_type}]\\n{status}"
        dot_lines.append(f'  "{node.node_id}" [label="{label}"];')

    for edge in edges:
        status = "" if edge.valid_to is None else f" ({edge.valid_to})"
        label = f"{edge.rel_label}{status}"
        dot_lines.append(f'  "{edge.source}" -> "{edge.target}" [label="{label}"];')

    dot_lines.append("}")
    dot_path = dot_path.resolve()
    dot_path.parent.mkdir(parents=True, exist_ok=True)
    dot_path.write_text("\n".join(dot_lines))

    if render_png:
        dot_bin = shutil.which("dot")
        output_path = png_path or dot_path.with_suffix(".png")
        if dot_bin:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([dot_bin, "-Tpng", str(dot_path), "-o", str(output_path)], check=False)
    return dot_path
