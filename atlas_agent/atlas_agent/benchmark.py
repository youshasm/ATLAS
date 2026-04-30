from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Any

from .config import AtlasConfig, default_config
from .core import AtlasAgent
from .core import Planner, Caller, Summarizer, Verifier, _infer_preferred_tools, build_arithmetic_expression
from .schemas import Plan, PlanStep
from .utils.token_count import count_tokens


BENCHMARK_CASES: list[dict[str, Any]] = [
    {
        "id": "math_add_easy",
        "query": "Find the total of 12 and 30",
        "expected_primary_tool": "calculator",
        "expected_substring": "42",
    },
    {
        "id": "math_expr_hard",
        "query": "What is (17*23) - (19*5)?",
        "expected_primary_tool": "calculator",
        "expected_substring": "296",
    },
    {
        "id": "kb_roles",
        "query": "Search the local knowledge base for planner and summarizer roles",
        "expected_primary_tool": "knowledge_base",
        "expected_substring": "knowledge base lookup completed",
    },
    {
        "id": "note_codename",
        "query": "Please store this note: the project codename is Atlas-Phoenix.",
        "expected_primary_tool": "text",
    },
    {
        "id": "recall_codename",
        "query": "What is the project codename I told you earlier?",
        "expected_primary_tool": "text",
        "expected_substring": "Atlas-Phoenix",
    },
    {
        "id": "verifier_catches_tool_error",
        "query": "Evaluate 5//2 + 1",
        "expected_primary_tool": "calculator",
        "expected_verified": False,
        "expected_substring": "Unsupported operator",
    },
    {
        "id": "web_search_1",
        "query": "Use the web_search tool to search the web for: what does JSONL mean? Then summarize.",
        "expected_primary_tool": "web_search",
        "expected_substring": "Stub result for:",
    },
    {
        "id": "web_search_2",
        "query": "Use the web_search tool to search the internet for: what is the purpose of PEP 8? Summarize in one sentence.",
        "expected_primary_tool": "web_search",
        "expected_substring": "Stub result for:",
    },
    {
        "id": "web_search_3",
        "query": "Use the web_search tool to search the web for: Ollama /api/generate format json. Then summarize.",
        "expected_primary_tool": "web_search",
        "expected_substring": "Stub result for:",
    },
]


@dataclass
class BenchmarkRow:
    variant: str
    backend: str
    memory_backend: str
    embedding_provider: str
    use_memory: bool
    use_verifier: bool
    case_id: str
    query: str
    expected_primary_tool: str
    expected_substring: str
    expected_verified: bool
    predicted_primary_tool: str
    primary_tool_match: bool
    verified: bool
    expected_substring_match: bool
    expected_verified_match: bool
    case_pass: bool
    step_count: int
    tool_calls: int
    coordination_mode: str
    retries: int
    latency_ms: float
    estimated_tokens: int


@dataclass
class VariantSummary:
    variant: str
    backend: str
    memory_backend: str
    embedding_provider: str
    use_memory: bool
    use_verifier: bool
    cases: int
    success_rate: float
    primary_tool_match_rate: float
    expected_substring_match_rate: float
    case_pass_rate: float
    avg_steps: float
    avg_tool_calls: float
    avg_retries: float
    iterative_mode_rate: float
    avg_latency_ms: float
    avg_estimated_tokens: float


def run_benchmark(
    backends: list[str],
    llm_model: str,
    llm_api_base: str,
    memory_backend: str = "hybrid",
    memory_vector_weight: float = 0.7,
    embedding_provider: str = "hash",
    embedding_api_base: str = "http://localhost:11434",
    embedding_model: str = "nomic-embed-text",
    include_no_memory: bool = True,
    include_no_verifier: bool = False,
    include_react_baseline: bool = False,
    include_decomposition_baseline: bool = False,
    cases: list[dict[str, Any]] | None = None,
    csv_path: Path | None = None,
    json_path: Path | None = None,
) -> dict[str, Any]:
    rows: list[BenchmarkRow] = []
    benchmark_cases = cases or BENCHMARK_CASES

    for backend in backends:
        full_config = default_config(
            llm_backend=backend,
            llm_model=llm_model,
            llm_api_base=llm_api_base,
            memory_backend=memory_backend,
            memory_vector_weight=memory_vector_weight,
            embedding_provider=embedding_provider,
            embedding_api_base=embedding_api_base,
            embedding_model=embedding_model,
            use_memory=True,
        )
        _clear_file(full_config.memory_path)
        rows.extend(_run_variant("full", full_config, benchmark_cases))

        if include_react_baseline:
            react_config = replace(full_config, use_memory=False, use_verifier=False)
            _clear_file(react_config.memory_path)
            rows.extend(_run_react_baseline_variant("react_baseline", react_config, benchmark_cases))

        if include_decomposition_baseline:
            decomp_config = replace(full_config, use_memory=False, use_verifier=False)
            _clear_file(decomp_config.memory_path)
            rows.extend(_run_decomposition_baseline_variant("decomposition_baseline", decomp_config, benchmark_cases))

        if include_no_memory:
            no_mem_config = replace(full_config, use_memory=False)
            _clear_file(no_mem_config.memory_path)
            rows.extend(_run_variant("no_memory", no_mem_config, benchmark_cases))

        if include_no_verifier:
            no_verifier_config = replace(full_config, use_verifier=False)
            _clear_file(no_verifier_config.memory_path)
            rows.extend(_run_variant("no_verifier", no_verifier_config, benchmark_cases))

    summaries = _summarize(rows)
    payload = {
        "summaries": [asdict(item) for item in summaries],
        "rows": [asdict(item) for item in rows],
    }

    if csv_path is not None:
        _write_csv(rows, csv_path)
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def _run_variant(
    variant: str,
    config: AtlasConfig,
    cases: list[dict[str, Any]],
    forced_mode: str | None = None,
) -> list[BenchmarkRow]:
    agent = AtlasAgent(config)
    out: list[BenchmarkRow] = []
    for case in cases:
        result = agent.run(case["query"], forced_mode=forced_mode)
        plan = result.get("plan", [])
        predicted_primary_tool = plan[0]["tool_name"] if plan else ""
        tool_calls = len(result.get("results", []))
        verified = bool(result.get("verified", False))
        coordination_mode = str(result.get("coordination_mode", ""))
        retries = int(result.get("retries", 0))
        latency_ms = float(result.get("latency_ms", 0.0))
        estimated_tokens = int(result.get("estimated_tokens", 0))
        expected_substring = str(case.get("expected_substring", ""))
        expected_verified = bool(case.get("expected_verified", True))
        summary_text = str(result.get("summary", ""))
        expected_substring_match = True if not expected_substring else (expected_substring in summary_text)
        expected_verified_match = verified == expected_verified
        primary_tool_match = predicted_primary_tool == case["expected_primary_tool"]
        case_pass = primary_tool_match and expected_substring_match and expected_verified_match
        out.append(
            BenchmarkRow(
                variant=variant,
                backend=config.llm_backend,
                memory_backend=config.memory_backend,
                embedding_provider=config.embedding_provider,
                use_memory=config.use_memory,
                use_verifier=config.use_verifier,
                case_id=case["id"],
                query=case["query"],
                expected_primary_tool=case["expected_primary_tool"],
                expected_substring=expected_substring,
                expected_verified=expected_verified,
                predicted_primary_tool=predicted_primary_tool,
                primary_tool_match=primary_tool_match,
                verified=verified,
                expected_substring_match=expected_substring_match,
                expected_verified_match=expected_verified_match,
                case_pass=case_pass,
                step_count=len(plan),
                tool_calls=tool_calls,
                coordination_mode=coordination_mode,
                retries=retries,
                latency_ms=latency_ms,
                estimated_tokens=estimated_tokens,
            )
        )
    return out


def _summarize(rows: list[BenchmarkRow]) -> list[VariantSummary]:
    groups: dict[tuple[str, str, str, str, bool, bool], list[BenchmarkRow]] = {}
    for row in rows:
        key = (
            row.variant,
            row.backend,
            row.memory_backend,
            row.embedding_provider,
            row.use_memory,
            row.use_verifier,
        )
        groups.setdefault(key, []).append(row)

    summaries: list[VariantSummary] = []
    for (variant, backend, memory_backend, embedding_provider, use_memory, use_verifier), items in groups.items():
        summaries.append(
            VariantSummary(
                variant=variant,
                backend=backend,
                memory_backend=memory_backend,
                embedding_provider=embedding_provider,
                use_memory=use_memory,
                use_verifier=use_verifier,
                cases=len(items),
                success_rate=_safe_rate(sum(1 for item in items if item.verified), len(items)),
                primary_tool_match_rate=_safe_rate(sum(1 for item in items if item.primary_tool_match), len(items)),
                expected_substring_match_rate=_safe_rate(
                    sum(1 for item in items if item.expected_substring_match),
                    len(items),
                ),
                case_pass_rate=_safe_rate(sum(1 for item in items if item.case_pass), len(items)),
                avg_steps=mean(item.step_count for item in items),
                avg_tool_calls=mean(item.tool_calls for item in items),
                avg_retries=mean(item.retries for item in items),
                iterative_mode_rate=_safe_rate(
                    sum(1 for item in items if item.coordination_mode == "iterative_plan"),
                    len(items),
                ),
                avg_latency_ms=mean(item.latency_ms for item in items),
                avg_estimated_tokens=mean(item.estimated_tokens for item in items),
            )
        )
    return sorted(summaries, key=lambda item: (item.backend, item.memory_backend, item.embedding_provider, item.variant))


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _write_csv(rows: list[BenchmarkRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _clear_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark cases file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [_normalize_case(item, idx) for idx, item in enumerate(payload)]
        if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
            return [_normalize_case(item, idx) for idx, item in enumerate(payload["cases"])]
        raise ValueError("JSON benchmark file must be a list or an object with a 'cases' list")

    if suffix in {".jsonl", ".ndjson"}:
        out: list[dict[str, Any]] = []
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            out.append(_normalize_case(json.loads(line), idx))
        return out

    raise ValueError("Unsupported cases format. Use .json, .jsonl, or .ndjson")


def _normalize_case(raw: Any, idx: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Case at index {idx} must be an object")

    query = str(raw.get("query", "")).strip()
    if not query:
        raise ValueError(f"Case at index {idx} is missing a non-empty 'query'")

    expected_primary_tool = str(raw.get("expected_primary_tool", "text")).strip() or "text"
    expected_substring = str(raw.get("expected_substring", ""))
    expected_verified = bool(raw.get("expected_verified", True))
    case_id = str(raw.get("id", f"case_{idx + 1}"))

    return {
        "id": case_id,
        "query": query,
        "expected_primary_tool": expected_primary_tool,
        "expected_substring": expected_substring,
        "expected_verified": expected_verified,
    }


def _run_react_baseline_variant(variant: str, config: AtlasConfig, cases: list[dict[str, Any]]) -> list[BenchmarkRow]:
    caller = Caller()
    summarizer = Summarizer()
    verifier = Verifier()
    out: list[BenchmarkRow] = []
    for case in cases:
        import time

        started = time.perf_counter()
        query = case["query"]
        # pick a single primary tool based on simple heuristics
        preferred = _infer_preferred_tools(query)
        if preferred:
            primary = next(iter(preferred))
        else:
            primary = "text"

        # build a single-step plan
        if primary == "calculator":
            expr = build_arithmetic_expression(query)
            step = PlanStep(id="step-1", tool_name="calculator", arguments={"expression": expr}, purpose="ReAct baseline direct call")
        elif primary == "web_search":
            step = PlanStep(id="step-1", tool_name="web_search", arguments={"query": query}, purpose="ReAct baseline direct call")
        elif primary == "knowledge_base":
            step = PlanStep(id="step-1", tool_name="knowledge_base", arguments={"query": query}, purpose="ReAct baseline direct call")
        else:
            step = PlanStep(id="step-1", tool_name="text", arguments={"message": query}, purpose="ReAct baseline direct call")

        plan = Plan(query=query, steps=[step], notes="ReAct baseline (direct)")
        result = caller.call(step)
        results = [result]

        if config.use_verifier:
            verified, verify_reason = verifier.verify(plan, results)
        else:
            verified, verify_reason = True, "Verifier disabled"

        summary = summarizer.summarize(query, plan, results, verified, verify_reason, 0)
        latency = (time.perf_counter() - started) * 1000.0
        est_tokens = count_tokens(query, config.llm_model) + count_tokens(summary, config.llm_model)

        predicted_primary_tool = plan.steps[0].tool_name if plan.steps else ""
        tool_calls = len(results)

        expected_substring = str(case.get("expected_substring", ""))
        expected_verified = bool(case.get("expected_verified", True))
        expected_substring_match = True if not expected_substring else (expected_substring in summary)
        expected_verified_match = verified == expected_verified
        primary_tool_match = predicted_primary_tool == case["expected_primary_tool"]
        case_pass = primary_tool_match and expected_substring_match and expected_verified_match

        out.append(
            BenchmarkRow(
                variant=variant,
                backend=config.llm_backend,
                memory_backend=config.memory_backend,
                embedding_provider=config.embedding_provider,
                use_memory=config.use_memory,
                use_verifier=config.use_verifier,
                case_id=case["id"],
                query=query,
                expected_primary_tool=case["expected_primary_tool"],
                expected_substring=expected_substring,
                expected_verified=expected_verified,
                predicted_primary_tool=predicted_primary_tool,
                primary_tool_match=primary_tool_match,
                verified=verified,
                expected_substring_match=expected_substring_match,
                expected_verified_match=expected_verified_match,
                case_pass=case_pass,
                step_count=len(plan.steps),
                tool_calls=tool_calls,
                coordination_mode="direct_call",
                retries=0,
                latency_ms=latency,
                estimated_tokens=est_tokens,
            )
        )

    return out


def _run_decomposition_baseline_variant(variant: str, config: AtlasConfig, cases: list[dict[str, Any]]) -> list[BenchmarkRow]:
    planner = Planner(config)
    caller = Caller()
    summarizer = Summarizer()
    verifier = Verifier()
    out: list[BenchmarkRow] = []
    for case in cases:
        import time

        started = time.perf_counter()
        query = case["query"]
        # Plan using planner (no memory, no verifier in variant config)
        plan = planner.plan(query, memory_context=[], mode="single_plan")
        results = [caller.call(step) for step in plan.steps]

        if config.use_verifier:
            verified, verify_reason = verifier.verify(plan, results)
        else:
            verified, verify_reason = True, "Verifier disabled"

        summary = summarizer.summarize(query, plan, results, verified, verify_reason, 0)
        latency = (time.perf_counter() - started) * 1000.0
        est_tokens = count_tokens(query, config.llm_model) + count_tokens(summary, config.llm_model)

        predicted_primary_tool = plan.steps[0].tool_name if plan.steps else ""
        tool_calls = len(results)

        expected_substring = str(case.get("expected_substring", ""))
        expected_verified = bool(case.get("expected_verified", True))
        expected_substring_match = True if not expected_substring else (expected_substring in summary)
        expected_verified_match = verified == expected_verified
        primary_tool_match = predicted_primary_tool == case["expected_primary_tool"]
        case_pass = primary_tool_match and expected_substring_match and expected_verified_match

        out.append(
            BenchmarkRow(
                variant=variant,
                backend=config.llm_backend,
                memory_backend=config.memory_backend,
                embedding_provider=config.embedding_provider,
                use_memory=config.use_memory,
                use_verifier=config.use_verifier,
                case_id=case["id"],
                query=query,
                expected_primary_tool=case["expected_primary_tool"],
                expected_substring=expected_substring,
                expected_verified=expected_verified,
                predicted_primary_tool=predicted_primary_tool,
                primary_tool_match=primary_tool_match,
                verified=verified,
                expected_substring_match=expected_substring_match,
                expected_verified_match=expected_verified_match,
                case_pass=case_pass,
                step_count=len(plan.steps),
                tool_calls=tool_calls,
                coordination_mode="single_plan",
                retries=0,
                latency_ms=latency,
                estimated_tokens=est_tokens,
            )
        )

    return out
