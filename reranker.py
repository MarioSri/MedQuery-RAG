"""Cross-encoder reranker (roadmap item 6).

Retrieves candidates -> scores (query, candidate) pairs -> re-ranks -> top N.
Starts with one configurable reranker per the roadmap's advice
("You don't need a complicated cross-encoder infrastructure immediately.
Start with one configurable reranker.").
"""
import numpy as np

import config


class Reranker:
    def rerank(self, query: str, candidates: list) -> list:
        """candidates: list of (chunk_id, content, score)
        Returns: list of (chunk_id, content, new_score) sorted best-first."""
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    """Sentence-transformers cross-encoder (e.g. ms-marco-MiniLM-L-6-v2)."""

    def __init__(self):
        from sentence_transformers import CrossEncoder  # part of sentence-transformers
        self._model = CrossEncoder(config.RERANK_MODEL)

    def rerank(self, query, candidates):
        if not candidates:
            return []
        pairs = [(query, content) for _, content, _ in candidates]
        scores = self._model.predict(pairs)
        scored = [
            (cid, content, float(s))
            for (cid, content, _), s in zip(candidates, scores)
        ]
        scored.sort(key=lambda t: -t[2])
        return scored


class NoopReranker(Reranker):
    """Passthrough — keeps the retrieval order (useful for evaluation baselines)."""

    def rerank(self, query, candidates):
        return list(candidates)


def get_reranker() -> Reranker:
    name = config.RERANKER.lower()
    if name in ("none", "off", "false", ""):
        return NoopReranker()
    if name == "cross_encoder":
        return CrossEncoderReranker()
    raise ValueError(f"Unknown reranker: {name}. Use 'none' or 'cross_encoder'.")
