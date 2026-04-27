from __future__ import annotations

import ast
import operator as op
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Tool:
    name: str
    description: str

    def run(self, **kwargs: Any) -> Any:
        raise NotImplementedError


class CalculatorTool(Tool):
    def __init__(self) -> None:
        super().__init__(name="calculator", description="Evaluate safe arithmetic expressions")

    def run(self, **kwargs: Any) -> Any:
        expression = str(kwargs.get("expression", ""))
        return safe_eval(expression)


class TextTool(Tool):
    def __init__(self) -> None:
        super().__init__(name="text", description="Return structured text guidance")

    def run(self, **kwargs: Any) -> Any:
        message = str(kwargs.get("message", ""))
        return {"message": message, "timestamp": datetime.now(timezone.utc).isoformat()}


class WebSearchStubTool(Tool):
    def __init__(self) -> None:
        super().__init__(name="web_search", description="Stub search tool for demonstration")

    def run(self, **kwargs: Any) -> Any:
        query = str(kwargs.get("query", ""))
        return {"query": query, "results": [f"Stub result for: {query}"]}


class KnowledgeBaseTool(Tool):
    def __init__(self, knowledge_file: Path) -> None:
        super().__init__(name="knowledge_base", description="Retrieve matching snippets from a local knowledge file")
        self.knowledge_file = knowledge_file

    def run(self, **kwargs: Any) -> Any:
        query = str(kwargs.get("query", ""))
        if not self.knowledge_file.exists():
            return {"query": query, "matches": []}

        query_tokens = set(tokenize(query))
        matches: list[str] = []
        for line in self.knowledge_file.read_text(encoding="utf-8").splitlines():
            line_tokens = set(tokenize(line))
            if not line_tokens:
                continue
            if query_tokens & line_tokens:
                matches.append(line)

        return {"query": query, "matches": matches[:5]}


SAFE_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.Mod: op.mod,
}


def safe_eval(expression: str) -> Any:
    node = ast.parse(expression, mode="eval")
    return _eval_node(node.body)


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed")
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        operator = SAFE_OPERATORS.get(type(node.op))
        if operator is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return operator(left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        operator = SAFE_OPERATORS.get(type(node.op))
        if operator is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return operator(operand)
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in text.replace("/", " ").replace("-", " ").replace("_", " ").split() if token]
