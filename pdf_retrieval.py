"""Hybrid lexical + embedding retrieval for local course PDF pages."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable
from typing import Any

log = logging.getLogger("pdf-retrieval")

_PAGE_EMBEDDINGS: dict[tuple[str, int, int], list[float]] = {}


def clear_embedding_cache() -> None:
    _PAGE_EMBEDDINGS.clear()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    an = math.sqrt(sum(x * x for x in a))
    bn = math.sqrt(sum(y * y for y in b))
    if not an or not bn:
        return 0.0
    return dot / (an * bn)


def _embed(client_factory: Callable[[], Any], texts: list[str]) -> list[list[float]]:
    model = os.environ.get("PDF_EMBED_MODEL", "grok-embedding-small")
    response = client_factory().embeddings.create(model=model, input=texts)
    return [list(item.embedding) for item in response.data]


def hybrid_rank(
    query: str,
    pages: list[dict],
    lexical_scores: list[float],
    client_factory: Callable[[], Any],
) -> tuple[list[tuple[float, dict]], str]:
    """Rank pages by lexical overlap plus xAI embeddings, with lexical fallback."""
    if not pages:
        return [], "lexical"

    lexical_max = max(lexical_scores, default=0.0)
    lexical_norm = [score / lexical_max if lexical_max else 0.0 for score in lexical_scores]
    if os.environ.get("PDF_SEMANTIC_RETRIEVAL", "1") == "0":
        ranked = sorted(zip(lexical_norm, pages), key=lambda item: item[0], reverse=True)
        return [(score, page) for score, page in ranked if score > 0], "lexical"

    try:
        missing: list[tuple[tuple[str, int, int], str]] = []
        keys: list[tuple[str, int, int]] = []
        for page in pages:
            text = str(page.get("text", ""))
            key = (str(page.get("file", "")), int(page.get("page", 0)), hash(text))
            keys.append(key)
            if key not in _PAGE_EMBEDDINGS:
                missing.append((key, text[:12000]))

        batch_size = max(1, int(os.environ.get("PDF_EMBED_BATCH", "32")))
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            vectors = _embed(client_factory, [text for _, text in batch])
            for (key, _), vector in zip(batch, vectors):
                _PAGE_EMBEDDINGS[key] = vector

        query_vector = _embed(client_factory, [query])[0]
        semantic = [max(0.0, _cosine(query_vector, _PAGE_EMBEDDINGS[key])) for key in keys]
        lexical_weight = float(os.environ.get("PDF_LEXICAL_WEIGHT", "0.4"))
        lexical_weight = min(max(lexical_weight, 0.0), 1.0)
        semantic_weight = 1.0 - lexical_weight
        combined = [
            lexical_weight * lexical + semantic_weight * sem
            for lexical, sem in zip(lexical_norm, semantic)
        ]
        ranked = sorted(zip(combined, pages), key=lambda item: item[0], reverse=True)
        return [(score, page) for score, page in ranked if score > 0], "hybrid"
    except Exception as exc:
        log.warning("semantic PDF retrieval unavailable; using lexical fallback: %s", exc)
        ranked = sorted(zip(lexical_norm, pages), key=lambda item: item[0], reverse=True)
        return [(score, page) for score, page in ranked if score > 0], "lexical"
