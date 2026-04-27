from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlanStep:
    id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    purpose: str = ""


@dataclass
class Plan:
    query: str
    steps: list[PlanStep] = field(default_factory=list)
    notes: str = ""


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: Any
    error: str | None = None


@dataclass
class MemoryItem:
    query: str
    plan_summary: str
    tool_sequence: list[str]
    result_summary: str
    success: bool
