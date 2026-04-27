from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

from .config import AtlasConfig
from .llm import choose_step_with_llm
from .memory import MemoryStore
from .schemas import MemoryItem, Plan, PlanStep, ToolResult
from .tools import CalculatorTool, KnowledgeBaseTool, TextTool, WebSearchStubTool


class Planner:
    def __init__(self, config: AtlasConfig) -> None:
        self.config = config

    def plan(
        self,
        query: str,
        memory_context: list[MemoryItem],
        mode: str = "single_plan",
        retry_context: str = "",
    ) -> Plan:
        lowered = query.lower()
        tokens = _tokenize(lowered)
        steps: list[PlanStep] = []
        notes = f"Coordination mode: {mode}."

        if memory_context:
            notes = notes + f" Retrieved {len(memory_context)} relevant memory item(s)."

        if retry_context:
            notes = notes + f" Recovery context: {retry_context}."

        if memory_context and _looks_like_memory_recall(tokens, lowered):
            recalled = _recall_from_memory(memory_context, query)
            steps.append(
                PlanStep(
                    id="step-1",
                    tool_name="text",
                    arguments={"message": recalled},
                    purpose="Answer using retrieved memory",
                )
            )
            return Plan(query=query, steps=steps, notes=notes)

        explicit = _explicit_tool_request(lowered, query)
        if explicit is not None:
            steps.append(
                PlanStep(
                    id="step-1",
                    tool_name=explicit["tool_name"],
                    arguments=explicit["arguments"],
                    purpose="Follow explicit user tool request",
                )
            )
            if notes:
                notes = notes + " Explicit tool request detected."
            else:
                notes = "Explicit tool request detected."
            # Continue below to allow adding follow-up steps (e.g., summarize web search) and memory refinement.

        llm_step = None
        if self.config.llm_backend != "rules":
            llm_step = choose_step_with_llm(
                backend=self.config.llm_backend,
                model=self.config.llm_model,
                api_base=self.config.llm_api_base,
                api_key_env=self.config.llm_api_key_env,
                query=query,
                memory_context=[asdict(item) for item in memory_context],
            )
            if llm_step:
                notes = (notes + " " if notes else "") + f"Planner used {self.config.llm_backend} backend."

        if steps:
            pass
        elif llm_step is not None:
            steps.append(
                PlanStep(
                    id="step-1",
                    tool_name=llm_step["tool_name"],
                    arguments=llm_step["arguments"],
                    purpose=llm_step["purpose"] or "LLM-selected step",
                )
            )
        elif _looks_like_arithmetic_query(query, lowered, tokens):
            steps.append(
                PlanStep(
                    id="step-1",
                    tool_name="calculator",
                    arguments={"expression": build_arithmetic_expression(query)},
                    purpose="Compute arithmetic result",
                )
            )
        elif set(["search", "find", "lookup", "look", "remember", "knowledge"]) & tokens:
            steps.append(
                PlanStep(
                    id="step-1",
                    tool_name="knowledge_base",
                    arguments={"query": query},
                    purpose="Find relevant information",
                )
            )
        else:
            steps.append(
                PlanStep(
                    id="step-1",
                    tool_name="text",
                    arguments={"message": f"No specialized tool selected for: {query}"},
                    purpose="Provide a fallback response",
                )
            )

        if mode != "direct_call" and steps and steps[0].tool_name == "web_search":
            steps.append(
                PlanStep(
                    id=f"step-{len(steps) + 1}",
                    tool_name="text",
                    arguments={"message": "Summarize the web search results clearly and briefly."},
                    purpose="Summarize search results",
                )
            )

        if mode != "direct_call" and memory_context:
            steps.append(
                PlanStep(
                    id=f"step-{len(steps) + 1}",
                    tool_name="text",
                    arguments={"message": "Use memory to refine the final answer."},
                    purpose="Incorporate past experience",
                )
            )

        return Plan(query=query, steps=steps, notes=notes)


def _explicit_tool_request(lowered: str, query: str) -> dict[str, Any] | None:
    if "web_search" in lowered or "web search" in lowered:
        return {"tool_name": "web_search", "arguments": {"query": query}}
    if "knowledge_base" in lowered or "knowledge base" in lowered:
        return {"tool_name": "knowledge_base", "arguments": {"query": query}}
    if "calculator" in lowered:
        return {"tool_name": "calculator", "arguments": {"expression": build_arithmetic_expression(query)}}
    return None


def _looks_like_memory_recall(tokens: set[str], lowered: str) -> bool:
    if "told you" in lowered or "did i" in lowered:
        return True
    if "?" not in lowered:
        return False
    recall_tokens = {
        "earlier",
        "previous",
        "before",
        "last",
        "recall",
        "again",
        "repeat",
        "codename",
    }
    return bool(tokens & recall_tokens)


def _looks_like_arithmetic_query(query: str, lowered: str, tokens: set[str]) -> bool:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", query)
    if len(numbers) < 2:
        return False

    if any(symbol in lowered for symbol in ["+", "-", "*", "/", "(", ")"]):
        return True

    arithmetic_tokens = {
        "sum",
        "add",
        "plus",
        "total",
        "subtract",
        "minus",
        "difference",
        "multiply",
        "times",
        "product",
        "divide",
        "quotient",
        "per",
    }
    return bool(tokens & arithmetic_tokens)


def _recall_from_memory(memory_context: list[MemoryItem], query: str) -> str:
    codename = _extract_codename(memory_context)
    if codename:
        return f"From memory, the project codename is {codename}."

    top = memory_context[0]
    return (
        "From memory, a related past interaction was: "
        f"query='{top.query}' plan='{top.plan_summary}'."
    )


def _extract_codename(memory_context: list[MemoryItem]) -> str:
    for item in memory_context:
        haystack = f"{item.query} {item.result_summary}"
        match = re.search(r"codename\s+is\s+([A-Za-z0-9_-]+)", haystack, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


class Caller:
    def __init__(self) -> None:
        knowledge_file = Path(__file__).resolve().parent / "data" / "sample_knowledge_base.txt"
        self.tools = {
            "calculator": CalculatorTool(),
            "text": TextTool(),
            "web_search": WebSearchStubTool(),
            "knowledge_base": KnowledgeBaseTool(knowledge_file),
        }

    def call(self, step: PlanStep) -> ToolResult:
        tool = self.tools.get(step.tool_name)
        if tool is None:
            return ToolResult(tool_name=step.tool_name, success=False, output=None, error="Unknown tool")
        try:
            output = tool.run(**step.arguments)
            return ToolResult(tool_name=step.tool_name, success=True, output=output)
        except Exception as exc:
            return ToolResult(tool_name=step.tool_name, success=False, output=None, error=str(exc))


class Summarizer:
    def summarize(
        self,
        query: str,
        plan: Plan,
        results: list[ToolResult],
        verified: bool,
        verification_reason: str,
        retries: int,
    ) -> str:
        parts = [f"Query: {query}"]
        if plan.notes:
            parts.append(plan.notes)
        for step, result in zip(plan.steps, results):
            if result.success:
                parts.append(f"{step.tool_name}: {result.output}")
            else:
                parts.append(f"{step.tool_name} failed: {result.error}")
        if results and results[0].success and plan.steps[0].tool_name == "calculator":
            parts.append(f"Final answer: {results[0].output}")
        elif results and results[0].success and plan.steps[0].tool_name == "knowledge_base":
            parts.append("Final answer: knowledge base lookup completed successfully.")
        else:
            parts.append("Final answer generated from available evidence.")
        parts.append(f"Verified: {verified}")
        if verification_reason:
            parts.append(f"Verification detail: {verification_reason}")
        if retries:
            parts.append(f"Recovery retries: {retries}")
        return " | ".join(parts)


class Verifier:
    def verify(self, plan: Plan, results: list[ToolResult]) -> tuple[bool, str]:
        if len(plan.steps) != len(results):
            return False, "Plan/result length mismatch"

        for step, result in zip(plan.steps, results):
            if not result.success:
                return False, f"{step.tool_name} execution failed: {result.error}"
            ok, reason = self._verify_shape(step.tool_name, result.output)
            if not ok:
                return False, f"{step.tool_name} output invalid: {reason}"

        return True, "All checks passed"

    def _verify_shape(self, tool_name: str, output: Any) -> tuple[bool, str]:
        if tool_name == "calculator":
            if isinstance(output, (int, float)):
                return True, ""
            return False, "calculator output must be numeric"

        if tool_name == "text":
            if isinstance(output, dict) and "message" in output:
                return True, ""
            return False, "text output must contain 'message'"

        if tool_name == "web_search":
            if isinstance(output, dict) and isinstance(output.get("results"), list):
                return True, ""
            return False, "web_search output must contain list 'results'"

        if tool_name == "knowledge_base":
            if isinstance(output, dict) and isinstance(output.get("matches"), list):
                return True, ""
            return False, "knowledge_base output must contain list 'matches'"

        return False, "unknown tool in verifier"


class AdaptiveCoordinator:
    def choose_mode(self, query: str) -> str:
        lowered = query.lower()
        tokens = _tokenize(lowered)

        if _explicit_tool_request(lowered, query) is not None and not _looks_complex(lowered, tokens):
            return "direct_call"

        if _looks_like_arithmetic_query(query, lowered, tokens) and not _looks_complex(lowered, tokens):
            return "direct_call"

        if _looks_complex(lowered, tokens):
            return "iterative_plan"

        return "single_plan"


class RecoveryPolicy:
    def attempt_repair(self, plan: Plan, results: list[ToolResult], reason: str) -> Plan | None:
        failed_index = _first_failed_index(results)
        if failed_index is None:
            return None

        failed_step = plan.steps[failed_index]
        failed_result = results[failed_index]

        if failed_step.tool_name not in {"calculator", "knowledge_base", "web_search", "text"}:
            repaired_steps = list(plan.steps)
            repaired_steps[failed_index] = PlanStep(
                id=failed_step.id,
                tool_name="text",
                arguments={"message": f"Recovered from unknown tool '{failed_step.tool_name}'."},
                purpose="Recovery fallback",
            )
            return Plan(query=plan.query, steps=repaired_steps, notes=f"{plan.notes} Repaired unknown tool.")

        # Do not auto-repair deterministic operator errors; verifier should surface them.
        if failed_result.error and "Unsupported operator" in failed_result.error:
            return None

        if failed_step.tool_name == "web_search" and failed_result.error:
            repaired_steps = list(plan.steps)
            repaired_steps[failed_index] = PlanStep(
                id=failed_step.id,
                tool_name="text",
                arguments={"message": "Web search failed, provide a safe fallback response."},
                purpose="Recovery fallback",
            )
            return Plan(query=plan.query, steps=repaired_steps, notes=f"{plan.notes} {reason}")

        return None


class AtlasAgent:
    def __init__(self, config: AtlasConfig) -> None:
        self.config = config
        self.memory = MemoryStore(config.memory_path)
        self.planner = Planner(config)
        self.caller = Caller()
        self.summarizer = Summarizer()
        self.coordinator = AdaptiveCoordinator()
        self.verifier = Verifier()
        self.recovery = RecoveryPolicy()

    def run(self, query: str) -> dict[str, Any]:
        mode = self.coordinator.choose_mode(query)
        preferred_tools = _infer_preferred_tools(query)
        memory_context = (
            self.memory.search(
                query,
                top_k=self.config.top_k_memory,
                preferred_tools=preferred_tools,
                backend=self.config.memory_backend,
                vector_weight=self.config.memory_vector_weight,
                embedding_provider=self.config.embedding_provider,
                embedding_api_base=self.config.embedding_api_base,
                embedding_model=self.config.embedding_model,
            )
            if self.config.use_memory
            else []
        )
        plan = self.planner.plan(query, memory_context, mode=mode)
        results = [self.caller.call(step) for step in plan.steps]

        if mode == "iterative_plan":
            plan, results = self._run_iterative_rounds(query, memory_context, plan, results)

        if self.config.use_verifier:
            verified, verify_reason = self.verifier.verify(plan, results)
        else:
            verified, verify_reason = True, "Verifier disabled"

        retries = 0
        while self.config.use_verifier and not verified and retries < self.config.max_retries:
            repaired = self.recovery.attempt_repair(plan, results, verify_reason)
            if repaired is None:
                replanned = self.planner.plan(query, memory_context, mode=mode, retry_context=verify_reason)
                if _same_plan_signature(plan, replanned):
                    break
                plan = replanned
                results = [self.caller.call(step) for step in plan.steps]
                verified, verify_reason = self.verifier.verify(plan, results)
                retries += 1
                continue
            plan = repaired
            results = [self.caller.call(step) for step in plan.steps]
            verified, verify_reason = self.verifier.verify(plan, results)
            retries += 1

        summary = self.summarizer.summarize(query, plan, results, verified, verify_reason, retries)

        if self.config.use_memory:
            self.memory.add(
                MemoryItem(
                    query=query,
                    plan_summary=", ".join(step.tool_name for step in plan.steps),
                    tool_sequence=[step.tool_name for step in plan.steps],
                    result_summary=summary,
                    success=verified,
                )
            )

        record = {
            "query": query,
            "plan": [asdict(step) for step in plan.steps],
            "results": [asdict(result) for result in results],
            "verified": verified,
            "verification_reason": verify_reason,
            "coordination_mode": mode,
            "retries": retries,
            "summary": summary,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return record

    def _run_iterative_rounds(
        self,
        query: str,
        memory_context: list[MemoryItem],
        plan: Plan,
        results: list[ToolResult],
    ) -> tuple[Plan, list[ToolResult]]:
        iteration = 0
        current_plan = plan
        current_results = results

        while iteration < max(1, self.config.max_retries):
            verify_ok, verify_reason = self.verifier.verify(current_plan, current_results)
            if verify_ok:
                break

            retry_context = f"iter={iteration + 1}: {verify_reason}"
            candidate_plan = self.planner.plan(query, memory_context, mode="iterative_plan", retry_context=retry_context)
            if _same_plan_signature(current_plan, candidate_plan):
                break

            current_plan = candidate_plan
            current_results = [self.caller.call(step) for step in current_plan.steps]
            iteration += 1

        return current_plan, current_results


def build_arithmetic_expression(query: str) -> str:
    numbers = [match.group() for match in re.finditer(r"-?\d+(?:\.\d+)?", query)]
    lowered = query.lower()

    if len(numbers) >= 2:
        if any(word in lowered for word in ["add", "sum", "plus", "total"]):
            return " + ".join(numbers[:2])
        if any(word in lowered for word in ["subtract", "minus", "difference"]):
            return f"{numbers[0]} - {numbers[1]}"
        if any(word in lowered for word in ["multiply", "times", "product"]):
            return f"{numbers[0]} * {numbers[1]}"
        if any(word in lowered for word in ["divide", "quotient", "per"]):
            return f"{numbers[0]} / {numbers[1]}"

    compact = "".join(char for char in query if char in "0123456789+-*/(). ")
    return compact.strip() or "0"


def _tokenize(text: str) -> set[str]:
    return set(token for token in re.findall(r"[a-zA-Z]+", text) if token)


def _looks_complex(lowered: str, tokens: set[str]) -> bool:
    complexity_markers = {
        "then",
        "after",
        "first",
        "second",
        "third",
        "multi",
        "multiple",
        "compare",
        "analyze",
        "steps",
        "summarize",
    }
    if len(tokens) >= 16:
        return True
    return bool(tokens & complexity_markers) or (";" in lowered)


def _first_failed_index(results: list[ToolResult]) -> int | None:
    for idx, result in enumerate(results):
        if not result.success:
            return idx
    return None


def _infer_preferred_tools(query: str) -> set[str]:
    lowered = query.lower()
    tools: set[str] = set()
    if any(token in lowered for token in ["add", "sum", "plus", "minus", "times", "divide", "calculator", "+", "-", "*", "/"]):
        tools.add("calculator")
    if any(token in lowered for token in ["web", "internet", "search", "lookup"]):
        tools.add("web_search")
    if any(token in lowered for token in ["knowledge base", "local", "kb"]):
        tools.add("knowledge_base")
    if any(token in lowered for token in ["remember", "store", "note", "recall"]):
        tools.add("text")
    return tools


def _same_plan_signature(left: Plan, right: Plan) -> bool:
    left_sig = [(step.tool_name, sorted(step.arguments.items())) for step in left.steps]
    right_sig = [(step.tool_name, sorted(step.arguments.items())) for step in right.steps]
    return left_sig == right_sig
