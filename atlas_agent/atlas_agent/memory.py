from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .schemas import MemoryItem


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def add(self, item: MemoryItem) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    def load_all(self) -> list[MemoryItem]:
        items: list[MemoryItem] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                items.append(MemoryItem(**payload))
        return items

    def search(
        self,
        query: str,
        top_k: int = 3,
        preferred_tools: set[str] | None = None,
        backend: str = "hybrid",
        vector_weight: float = 0.7,
        embedding_provider: str = "hash",
        embedding_api_base: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
    ) -> list[MemoryItem]:
        items = self.load_all()
        vector_weight = min(1.0, max(0.0, vector_weight))

        scored = sorted(
            items,
            key=lambda item: hybrid_similarity_score(
                query=query,
                text=item.query + " " + item.plan_summary + " " + item.result_summary,
                tool_sequence=item.tool_sequence,
                preferred_tools=preferred_tools,
                backend=backend,
                vector_weight=vector_weight,
                embedding_provider=embedding_provider,
                embedding_api_base=embedding_api_base,
                embedding_model=embedding_model,
            ),
            reverse=True,
        )
        return scored[:top_k]


def similarity_score(query: str, text: str) -> float:
    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    union = len(query_tokens | text_tokens)
    return overlap / union


def hybrid_similarity_score(
    query: str,
    text: str,
    tool_sequence: Iterable[str],
    preferred_tools: set[str] | None,
    backend: str,
    vector_weight: float,
    embedding_provider: str,
    embedding_api_base: str,
    embedding_model: str,
) -> float:
    semantic = similarity_score(query, text)
    vector = vector_similarity_score(
        query,
        text,
        provider=embedding_provider,
        ollama_api_base=embedding_api_base,
        ollama_model=embedding_model,
    )

    if backend == "lexical":
        base_score = semantic
    elif backend == "vector":
        base_score = vector
    else:
        base_score = ((1.0 - vector_weight) * semantic) + (vector_weight * vector)

    if not preferred_tools:
        return base_score

    used_tools = {tool.lower() for tool in tool_sequence}
    if not used_tools:
        return base_score

    tool_overlap = len(used_tools & preferred_tools) / len(used_tools | preferred_tools)
    return (0.8 * base_score) + (0.2 * tool_overlap)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in text.replace("/", " ").replace("-", " ").split() if token]


def vector_similarity_score(
    query: str,
    text: str,
    provider: str = "hash",
    ollama_api_base: str = "http://localhost:11434",
    ollama_model: str = "nomic-embed-text",
) -> float:
    query_vec = embed_text(
        query,
        provider=provider,
        ollama_api_base=ollama_api_base,
        ollama_model=ollama_model,
    )
    text_vec = embed_text(
        text,
        provider=provider,
        ollama_api_base=ollama_api_base,
        ollama_model=ollama_model,
    )
    return cosine_similarity(query_vec, text_vec)


def embed_text(
    text: str,
    dim: int = 256,
    provider: str = "hash",
    ollama_api_base: str = "http://localhost:11434",
    ollama_model: str = "nomic-embed-text",
) -> list[float]:
    if provider == "ollama":
        embedded = _embed_with_ollama(text, ollama_api_base=ollama_api_base, ollama_model=ollama_model)
        if embedded:
            return embedded

    # Hashing-based embeddings are deterministic and lightweight for local vector retrieval.
    vec = [0.0] * dim
    for token in tokenize(text):
        h = hash(token)
        idx = abs(h) % dim
        sign = 1.0 if h >= 0 else -1.0
        weight = 1.0 + min(len(token), 12) / 12.0
        vec[idx] += sign * weight
    norm = math.sqrt(sum(value * value for value in vec))
    if norm == 0.0:
        return vec
    return [value / norm for value in vec]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def _embed_with_ollama(text: str, ollama_api_base: str, ollama_model: str) -> list[float]:
    url = ollama_api_base.rstrip("/") + "/api/embeddings"
    payload = json.dumps({"model": ollama_model, "prompt": text}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return []

    embedding = raw.get("embedding")
    if not isinstance(embedding, list):
        return []

    values: list[float] = []
    for item in embedding:
        if isinstance(item, (int, float)):
            values.append(float(item))
        else:
            return []

    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return values
    return [value / norm for value in values]
