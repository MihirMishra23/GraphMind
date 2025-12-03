# GraphMind

GraphMind is a novel way to augment LLMs with graph-based memory.

## Setup
```
conda create -n graphmind python=3.12
git clone https://github.com/MihirMishra23/GraphMind.git
cd GraphMind
pip install -e .
pip install jericho transformers torch accelerate graphviz
```
- Ensure the Jericho Z-Machine game assets are present (defaults to `data/jericho/z-machine-games-master/jericho-game-suite/zork1.z5`). If missing, download the Jericho game suite and drop the `.z5` under `data/jericho/`.
- (Optional) For GraphViz PNG export, install the `dot` binary (e.g., `brew install graphviz`).
- For the LLM agent, make sure you have access to the Hugging Face model id you pass (default: `meta-llama/Llama-3.2-3B-Instruct`) and that your machine has enough VRAM/CPU RAM.

## Quickstart commands
- Smoke-test the memory pipeline with the heuristic extractor (fast, no LLM load):  
  `python scripts/jericho_memory_smoke.py --max-steps 30 --graphviz-dot out/jericho_memory.dot --graphviz-png out/jericho_memory.png`
- Use an existing LLM extraction completion instead of the heuristic:  
  `python scripts/jericho_memory_smoke.py --use-llm-output path/to/completion.txt --quiet`
- Run a walkthrough-only Jericho episode:  
  `python scripts/jericho_runner.py --agent walkthrough --json-log out/walkthrough.json --text-log out/walkthrough.log`
- Run with the LLM agent + graph memory enabled (loads HF model):  
  `python scripts/jericho_runner.py --agent llm --memory-mode llama --model-id meta-llama/Llama-3.2-3B-Instruct --graphviz-dot out/run.dot --graphviz-png out/run.png --max-steps 200`
- Use `--game <path>` to swap games, `--quiet` to suppress per-step prints, and `--log-level DEBUG` for verbose internals.

## Architecture (graph-augmented LLM loop)
- **Environment + agent:** Jericho `FrotzEnv` emits observations; agents are either a fixed walkthrough or an LLM policy (`LLMAgent`) with optional memory conditioning.
- **Extraction:** Each observation is distilled into structured `entities`, `relations`, and `events` via either a heuristic parser (`naive_extract`) or an LLM extraction prompt (`extraction_mode=llm`), with tunable max tokens/temperature.
- **Grounding + graph store:** The `Grounder` merges new facts into a temporal graph (`GraphStore`), reopening/updating nodes/edges, attaching aliases/properties, and tracking active vs. closed facts.
- **Retrieval/conditioning:** When `memory-mode=llama`, the LLM agent pulls the grounded graph context (salient nodes/edges/events) into the prompt to steer action generation; when disabled, it behaves like a vanilla LLM policy.
- **Exports + logs:** Runs can emit text/JSON trajectories and GraphViz DOT/PNG snapshots of the evolving memory for inspection.
