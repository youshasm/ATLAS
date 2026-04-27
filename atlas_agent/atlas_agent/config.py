from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class AtlasConfig:
    project_root: Path
    memory_path: Path
    log_path: Path
    llm_backend: Literal["rules", "ollama", "online"] = "rules"
    llm_model: str = "llama3.1:8b"
    llm_api_base: str = "http://localhost:11434"
    llm_api_key_env: str = "OPENAI_API_KEY"
    memory_backend: Literal["lexical", "vector", "hybrid"] = "hybrid"
    memory_vector_weight: float = 0.7
    embedding_provider: Literal["hash", "ollama"] = "hash"
    embedding_api_base: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    use_memory: bool = True
    use_verifier: bool = True
    max_retries: int = 2
    top_k_memory: int = 3


def default_config(
    llm_backend: Literal["rules", "ollama", "online"] = "rules",
    llm_model: str = "llama3.1:8b",
    llm_api_base: str = "http://localhost:11434",
    memory_backend: Literal["lexical", "vector", "hybrid"] = "hybrid",
    memory_vector_weight: float = 0.7,
    embedding_provider: Literal["hash", "ollama"] = "hash",
    embedding_api_base: str = "http://localhost:11434",
    embedding_model: str = "nomic-embed-text",
    use_memory: bool = True,
    use_verifier: bool = True,
) -> AtlasConfig:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return AtlasConfig(
        project_root=project_root,
        memory_path=data_dir / "memory.jsonl",
        log_path=data_dir / "runs.jsonl",
        llm_backend=llm_backend,
        llm_model=llm_model,
        llm_api_base=llm_api_base,
        memory_backend=memory_backend,
        memory_vector_weight=memory_vector_weight,
        embedding_provider=embedding_provider,
        embedding_api_base=embedding_api_base,
        embedding_model=embedding_model,
        use_memory=use_memory,
        use_verifier=use_verifier,
    )
