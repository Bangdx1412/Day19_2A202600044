"""Lightweight node embeddings for graph construction and seed retrieval.

The offline default uses deterministic hashed bag-of-words vectors. This keeps
the lab reproducible without external APIs while still satisfying the graph
pipeline requirement that nodes carry embeddings. If desired, the same storage
shape can be swapped for OpenAI embeddings later.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from config import EMBEDDING_DIM, NODE_EMBEDDINGS_PATH


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9$.-]+", text.casefold())


def text_to_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    vector = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return vector
    return [round(value / norm, 6) for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def build_node_texts(triples: list[dict[str, str]]) -> dict[str, str]:
    node_facts: dict[str, list[str]] = {}
    for triple in triples:
        subject = triple["subject"]
        relation = triple["relation"].replace("_", " ").lower()
        obj = triple["object"]
        node_facts.setdefault(subject, []).append(f"{subject} {relation} {obj}")
        node_facts.setdefault(obj, []).append(f"{subject} {relation} {obj}")
    return {
        node: ". ".join(sorted(set(facts)))
        for node, facts in sorted(node_facts.items())
    }


def build_node_embeddings(triples: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    node_texts = build_node_texts(triples)
    return {
        node: {
            "text": text,
            "embedding": text_to_embedding(f"{node}. {text}"),
        }
        for node, text in node_texts.items()
    }


def save_node_embeddings(
    embeddings: dict[str, dict[str, object]],
    path: str | Path = NODE_EMBEDDINGS_PATH,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(embeddings, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
