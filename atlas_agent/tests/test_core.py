from __future__ import annotations

import json
from pathlib import Path

from atlas_agent.config import AtlasConfig
from atlas_agent.core import AtlasAgent
from atlas_agent.schemas import Plan, PlanStep


def make_config(tmp_path: Path) -> AtlasConfig:
    return AtlasConfig(
        project_root=tmp_path,
        memory_path=tmp_path / "memory.jsonl",
        log_path=tmp_path / "runs.jsonl",
    )


def test_calculator_task(tmp_path: Path) -> None:
    agent = AtlasAgent(make_config(tmp_path))
    result = agent.run("Find the total of 12 and 30")
    assert result["verified"] is True
    assert "42" in result["summary"]
    assert result["coordination_mode"] in {"direct_call", "single_plan", "iterative_plan"}


def test_knowledge_base_task(tmp_path: Path) -> None:
    agent = AtlasAgent(make_config(tmp_path))
    result = agent.run("remember the planner and summarizer roles")
    assert result["verified"] is True
    assert "knowledge base" in result["summary"].lower()


def test_memory_persists_success(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AtlasAgent(config)
    first = agent.run("Please add 7 and 18")
    second = agent.run("Please add 7 and 18")
    assert first["verified"] is True
    assert second["verified"] is True
    assert config.memory_path.exists()
    assert config.memory_path.read_text(encoding="utf-8").strip() != ""


def test_failure_trace_persisted_when_memory_enabled(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AtlasAgent(config)
    result = agent.run("Evaluate 5//2 + 1")
    assert result["verified"] is False

    lines = [line for line in config.memory_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines, "Memory should store traces even for failed runs"
    last_item = json.loads(lines[-1])
    assert last_item["success"] is False


def test_iterative_replanning_uses_retry_context(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AtlasAgent(config)

    class StubPlanner:
        def __init__(self) -> None:
            self.retry_contexts: list[str] = []

        def plan(self, query: str, memory_context: list, mode: str = "single_plan", retry_context: str = "") -> Plan:
            self.retry_contexts.append(retry_context)
            if retry_context:
                return Plan(
                    query=query,
                    steps=[PlanStep(id="step-1", tool_name="text", arguments={"message": "Recovered"}, purpose="replan")],
                    notes="replanned",
                )
            return Plan(
                query=query,
                steps=[PlanStep(id="step-1", tool_name="unknown_tool", arguments={}, purpose="force retry")],
                notes="initial",
            )

    class StubCoordinator:
        def choose_mode(self, query: str) -> str:
            return "iterative_plan"

    stub_planner = StubPlanner()
    agent.planner = stub_planner  # type: ignore[assignment]
    agent.coordinator = StubCoordinator()  # type: ignore[assignment]

    result = agent.run("Do a complex multi-step task and summarize")
    assert result["verified"] is True
    assert any(context for context in stub_planner.retry_contexts[1:])
