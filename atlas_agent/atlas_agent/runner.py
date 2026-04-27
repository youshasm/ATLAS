from __future__ import annotations

from .config import default_config
from .core import AtlasAgent
from .logging_utils import append_jsonl


def run_task(task: str) -> dict:
    config = default_config()
    agent = AtlasAgent(config)
    result = agent.run(task)
    append_jsonl(config.log_path, result)
    return result
