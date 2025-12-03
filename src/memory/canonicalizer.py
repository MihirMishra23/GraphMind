from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Tuple

from .schema import NodeType, WorldKG


@dataclass
class CanonicalEntityRef:
    canonical_id: str
    node_type: str
    display_name: str


class Canonicalizer:
    def __init__(self, world_kg: WorldKG):
        self.world_kg = world_kg

    def canonicalize_entity_text(
        self, text: str, default_type: str, step: int
    ) -> CanonicalEntityRef:
        """
        Map a text span to an existing or new canonical entity id.

        Use simple heuristics:
        - Lowercasing and stripping punctuation.
        - String similarity against known node names.
        - For now, no heavy NLP; keep it readable and extensible.
        """
        cleaned_text = text.strip()
        norm_text = self._normalize(cleaned_text)
        target_tokens = self._tokenize(norm_text)

        # Validate/normalize type early.
        node_type = NodeType(default_type).value

        # Try to match existing nodes by name similarity.
        match_id, match_score = self._find_best_match(target_tokens)
        if match_id and match_score >= 0.6:
            display_name = self.world_kg.graph.nodes[match_id].get("name", match_id)
            self._add_alias(match_id, cleaned_text, step)
            return CanonicalEntityRef(
                canonical_id=match_id, node_type=self.world_kg.graph.nodes[match_id]["type"], display_name=display_name
            )

        # Create a new canonical id based on normalized text.
        base_id = norm_text.replace(" ", "_") or default_type.lower()
        canonical_id = self._dedupe_id(base_id)
        existed = canonical_id in self.world_kg.graph
        self.world_kg.add_or_get_entity(canonical_id, node_type, name=cleaned_text)
        self._add_alias(canonical_id, cleaned_text, step)
        if not existed:
            self.world_kg.graph.nodes[canonical_id]["last_updated_step"] = step

        return CanonicalEntityRef(
            canonical_id=canonical_id,
            node_type=node_type,
            display_name=cleaned_text,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _normalize(self, text: str) -> str:
        lowered = text.lower()
        lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered).strip()
        return lowered

    def _tokenize(self, text: str) -> list[str]:
        return [tok for tok in text.split(" ") if tok]

    def _similarity(self, tokens_a: Iterable[str], tokens_b: Iterable[str]) -> float:
        set_a, set_b = set(tokens_a), set(tokens_b)
        if not set_a or not set_b:
            return 0.0
        overlap = len(set_a & set_b)
        denom = max(len(set_a), len(set_b))
        return overlap / denom if denom else 0.0

    def _find_best_match(self, target_tokens: list[str]) -> Tuple[str | None, float]:
        best_id: str | None = None
        best_score = 0.0
        target_joined = " ".join(target_tokens)
        for node_id, data in self.world_kg.graph.nodes(data=True):
            for candidate in self._alias_candidates(data, fallback=node_id):
                norm = self._normalize(candidate)
                tokens = self._tokenize(norm)
                score = self._similarity(tokens, target_tokens)
                if score > best_score:
                    best_score = score
                    best_id = node_id
                # Fast path for exact match.
                if norm == target_joined and norm:
                    return node_id, 1.0
        return best_id, best_score

    def _dedupe_id(self, base_id: str) -> str:
        candidate = base_id
        counter = 2
        while candidate in self.world_kg.graph:
            candidate = f"{base_id}_{counter}"
            counter += 1
        return candidate

    def _alias_candidates(self, data: dict, fallback: str) -> list[str]:
        aliases = data.get("aliases") or []
        name = data.get("name", fallback)
        if name:
            aliases = [name] + aliases
        return aliases

    def _add_alias(self, node_id: str, alias: str, step: int) -> None:
        node = self.world_kg.graph.nodes[node_id]
        aliases = node.get("aliases")
        if aliases is None:
            aliases = []
            node["aliases"] = aliases
        # Avoid duplicates ignoring case/whitespace/punctuation.
        norm_alias = self._normalize(alias)
        existing_norms = {self._normalize(a) for a in aliases}
        if norm_alias and norm_alias not in existing_norms:
            aliases.append(alias)
            node["last_updated_step"] = step
