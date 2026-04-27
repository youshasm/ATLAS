from __future__ import annotations

from dataclasses import dataclass

from .config import AtlasConfig, default_config
from .core import AtlasAgent


@dataclass
class EvaluationResult:
    total: int
    success_count: int
    accuracy: float


DEFAULT_TASKS = [
    "Find the total of 12 and 30",
    "Please add 7 and 18",
    "Search for agentic AI planning",
]


def evaluate_agent(config: AtlasConfig | None = None) -> EvaluationResult:
    config = config or default_config()
    agent = AtlasAgent(config)
    successes = 0
    for task in DEFAULT_TASKS:
        result = agent.run(task)
        if result["verified"]:
            successes += 1
    total = len(DEFAULT_TASKS)
    return EvaluationResult(total=total, success_count=successes, accuracy=successes / total if total else 0.0)
