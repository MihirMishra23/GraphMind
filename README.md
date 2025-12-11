# GraphMind

Lightweight Jericho runner with a walkthrough agent and an LLM agent (no graph memory).

## Setup
```
conda create -n graphmind python=3.12
git clone https://github.com/MihirMishra23/GraphMind.git
cd GraphMind
pip install -e .
pip install jericho transformers torch accelerate graphviz
```
- Ensure the Jericho Z-Machine game assets are present (defaults to `data/jericho/z-machine-games-master/jericho-game-suite/zork1.z5`). If missing, download the Jericho game suite and drop the `.z5` under `data/jericho/`.
- (Optional) For GraphViz PNG export in the graph search script, install the `dot` binary (e.g., `brew install graphviz`).
- For the LLM agent, make sure you have access to the Hugging Face model id you pass (default: `meta-llama/Llama-3.2-3B-Instruct`) and that your machine has enough VRAM/CPU RAM.

## Quickstart commands
- Run a walkthrough-only Jericho episode:  
  `python scripts/jericho_runner.py --agent walkthrough --json-log out/walkthrough.json --text-log out/walkthrough.log`
- Run with the LLM agent (vanilla prompt-based policy):  
  `python scripts/jericho_runner.py --agent llm --model-id meta-llama/Llama-3.2-3B-Instruct --max-steps 200`
- Use `--game <path>` to swap games, `--quiet` to suppress per-step prints, and `--log-level DEBUG` for verbose internals.

## Architecture (simple loop)
- **Environment + agent:** Jericho `FrotzEnv` emits observations; agents are either a fixed walkthrough or an LLM policy (`LLMAgent`).
- **Action selection:** The LLM agent conditions on the latest observation and a short recent history window; there is no knowledge graph or structured memory.
- **Logging:** Runs can emit text/JSON trajectories for inspection. Graph search visualization lives in `scripts/jericho_graphsearch.py`.
