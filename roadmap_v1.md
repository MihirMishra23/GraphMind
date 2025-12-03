# GraphMind Roadmap

## Overview
GraphMind augments LLM agents with a property-graph memory and an RL-guided policy to decide when and how to write/update/delete information. v0 targets Jericho text games with LLaMA-based extraction and open-vocab relations; storage and latency are not yet optimized.

## Scope & Assumptions
- Focus on Jericho text games only for v0; add other benchmarks later.
- Property graph with open-vocab relations; keep provenance and validity intervals; no compaction yet.
- LLM-based IE (LLaMA) to extract entities/relations/events; no budget constraints.
- Preferences can be inferred from actions/observations; explicit user statements rare.

## Minimal Property Graph Schema
- **Nodes**: `Entity` (id, type guess, aliases, embedding, confidence), `Event` (type, participants, time, text_span), `Preference` (user_id, slot, value, validity [t_start, t_end)).
- **Edges**: typed open-vocab `REL` with raw `rel_label`, `rel_embedding`, `rel_cluster_id` (simple online clustering), `confidence`, `source_turn`, `valid_from`, `valid_to` (None=active), `provenance` (model, prompt), `evidence_count`.
- **System edges**: `alias_of`, `temporal_next` for hygiene. No hard deletes; close intervals or mark contradicted.

## 1) Memory Graph Implementation
- Implement `memory/graph_store.py`: property graph objects (Node/Edge), validity intervals (`valid_from`, `valid_to`), provenance, confidence, evidence_count, alias_of edges, open-vocab rel labels + embedding slot + rel_cluster_id.
- Add indexing: surface-form → node ids, embedding index (stub with FAISS/ann once available), recency index.
- Implement operations: add_node/edge, update (close interval + new edge), delete (mark contradicted/close interval), alias merge (union-find), active view vs history view helpers.

## 2) Extraction & Grounding
- Implement `memory/extraction.py`: LLaMA IE prompt + parser emitting entities/events/relations with confidence/time.
- Few-shot Jericho examples for prompt; include implicit preference hints.
- Grounding module: lexical + embedding matching; threshold τ; create new nodes if distance > τ; maintain alias clusters.

## 3) Policy Module
- Implement `memory/policy.py`: op selector with heuristic baseline (e.g., always ADD unless contradiction signals) and a stub RL training interface.
- Feature construction: query/goal embedding, retrieved subgraph summary, novelty counts, recency stats, contradiction flags.
- Action space: ADD, UPDATE (close interval + add), DELETE (mark contradicted), NOOP.

## 4) Retrieval
- Implement `memory/retrieval.py`: hybrid lexical+embedding search over active graph; k-hop expansion; optional historical fetch for past-tense queries.
- API to return context snippets for agent/LLM consumption.

## 5) Integrate Memory into Runner
- Wire `jericho_runner.py` to instantiate a real memory store instead of `NullMemory`.
- On each step: retrieve context → pass to agent/LLM prompts; after env step run extraction → grounding → policy → commit to memory.
- Logging: persist memory diffs per turn for debugging.

## 6) Agent Prompting & Safety
- Update LLMAgent prompts to include retrieved context and allowed actions list; add guardrails for empty/invalid outputs.
- Add configurable system prompt and stop tokens in CLI args.

## 7) Evaluation Harness
- Expand `scripts/eval_runner.py` to compute: task score, contradiction rate (active incompatible edges), memory size stats, retrieval hit rate (vs. hand labels on small set).
- Add YAML config (`configs/jericho_v0.yaml`) for paths, thresholds, prompt templates.
- Optionally add `notebooks/jericho_eval.ipynb` for trajectory analysis.

## 8) Testing
- Unit tests for graph_store operations (interval closing, alias merges), extraction parser, grounding thresholding, policy heuristic decisions, retrieval ranking.
- Smoke test for runner with mocked LLM and tiny Jericho env stub to ensure end-to-end loop works.

## 9) Packaging & Docs
- Update `pyproject.toml` dependencies (transformers, torch, jericho, faiss/candidate ANN, pydantic if used).
- README section describing v0 pipeline, how to run LLMAgent, device selection (cuda > mps > cpu).
- Add `.gitignore` entries for model caches and logs if missing.

## 10) Stretch / Next Iteration
- Relation clustering implementation (online centroid update).
- Preference drift handling with explicit validity intervals and conflict resolution rules.
- Add additional benchmarks (TextWorld/AlfWorld/LIGHT) once Jericho path is stable.
