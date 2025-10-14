# GraphMind

GraphMind is a novel way to augment LLMs with graph-based memory. This project provides 3 state-of-the-art improvements over existing methods:
1. Memory knowledge is stored in a property graph instead of a triple-store, enabling richer data representations more suited for LLM processing
2. GraphMind decides when to commit data to memory by using an RL-based policy
3. GraphMind decides what data to retrieve by using an LLM-based fuzzy graph search on the property graph memory
