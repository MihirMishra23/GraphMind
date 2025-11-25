#!/usr/bin/env python3
"""Lightweight evaluator for Jericho trajectories logged by jericho_runner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Jericho trajectory logs")
    parser.add_argument("log", type=Path, help="Path to JSON log produced by jericho_runner")
    return parser.parse_args()


def summarize(trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_reward = sum(step.get("reward", 0) for step in trajectory)
    steps = len(trajectory)
    done = any(step.get("done") for step in trajectory)
    final_info = trajectory[-1].get("info", {}) if trajectory else {}
    return {
        "steps": steps,
        "total_reward": total_reward,
        "finished": done,
        "final_info": final_info,
    }


def main() -> None:
    args = parse_args()
    data = json.loads(args.log.read_text())
    summary = summarize(data)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
