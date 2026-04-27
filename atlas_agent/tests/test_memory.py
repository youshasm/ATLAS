from __future__ import annotations

from pathlib import Path

from atlas_agent.memory import MemoryStore
from atlas_agent.memory import embed_text
from atlas_agent.schemas import MemoryItem


def test_vector_backend_returns_top_item(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.jsonl")
    store.add(
        MemoryItem(
            query="project codename",
            plan_summary="text",
            tool_sequence=["text"],
            result_summary="The codename is Atlas-Phoenix",
            success=True,
        )
    )
    store.add(
        MemoryItem(
            query="math addition",
            plan_summary="calculator",
            tool_sequence=["calculator"],
            result_summary="42",
            success=True,
        )
    )

    results = store.search(
        "what codename did I tell you",
        top_k=1,
        preferred_tools={"text"},
        backend="vector",
        vector_weight=0.8,
    )

    assert results
    assert "codename" in results[0].query


def test_hybrid_backend_prefers_tool_overlap(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.jsonl")
    store.add(
        MemoryItem(
            query="find multiplication result",
            plan_summary="calculator",
            tool_sequence=["calculator"],
            result_summary="84",
            success=True,
        )
    )
    store.add(
        MemoryItem(
            query="search online reference",
            plan_summary="web_search",
            tool_sequence=["web_search"],
            result_summary="Stub result for: reference",
            success=True,
        )
    )

    results = store.search(
        "search for reference on the internet",
        top_k=1,
        preferred_tools={"web_search"},
        backend="hybrid",
        vector_weight=0.7,
    )

    assert results
    assert "web_search" in results[0].tool_sequence


def test_ollama_embedding_fallback_to_hash() -> None:
    text = "atlas memory recall"
    vector = embed_text(
        text,
        provider="ollama",
        ollama_api_base="http://127.0.0.1:9",
        ollama_model="nomic-embed-text",
    )
    assert vector
    assert isinstance(vector[0], float)
