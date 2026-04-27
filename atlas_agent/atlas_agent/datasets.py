from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any


ALLOWED_TOOLS = {"calculator", "knowledge_base", "web_search", "text"}


def convert_external_dataset(
    source_path: Path,
    out_path: Path,
    source_format: str = "auto",
    limit: int = 0,
    seed: int = 42,
) -> list[dict[str, Any]]:
    records = _load_records(source_path)
    cases = [_normalize_external_record(record, idx, source_format) for idx, record in enumerate(records)]
    cases = [case for case in cases if case is not None]

    if limit > 0 and len(cases) > limit:
        rng = random.Random(seed)
        cases = rng.sample(cases, k=limit)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    return cases


def _load_records(source_path: Path) -> list[Any]:
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {source_path}")

    suffix = source_path.suffix.lower()
    raw_text = source_path.read_text(encoding="utf-8")

    if suffix == ".json":
        payload = json.loads(raw_text)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ["data", "examples", "items", "records", "cases"]:
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        raise ValueError("Unsupported JSON structure; expected a list or object containing a list")

    if suffix in {".jsonl", ".ndjson"}:
        out: list[Any] = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
        return out

    raise ValueError("Unsupported file format. Use .json, .jsonl, or .ndjson")


def _normalize_external_record(record: Any, idx: int, source_format: str) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None

    query = _extract_query(record, source_format)
    if not query:
        return None

    expected_tool = _extract_tool(record)
    expected_substring = _extract_expected_substring(record)
    expected_verified = bool(record.get("expected_verified", True))
    case_id = str(record.get("id") or record.get("uid") or record.get("qid") or f"ext_case_{idx + 1}")

    return {
        "id": case_id,
        "query": query,
        "expected_primary_tool": expected_tool,
        "expected_substring": expected_substring,
        "expected_verified": expected_verified,
    }


def _extract_query(record: dict[str, Any], source_format: str) -> str:
    if source_format == "apibank":
        raw_input = record.get("input")
        if isinstance(raw_input, str):
            parsed = _extract_apibank_user_query(raw_input)
            if parsed:
                return parsed

        for key in ["instruction", "question", "query", "task"]:
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    if source_format in {"toolbench"}:
        for key in ["instruction", "query", "question", "task", "input"]:
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    for key in ["query", "instruction", "question", "user_query", "task", "prompt", "input", "text"]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    messages = record.get("messages")
    if isinstance(messages, list):
        for item in reversed(messages):
            if isinstance(item, dict) and item.get("role") == "user":
                content = item.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    return ""


def _extract_tool(record: dict[str, Any]) -> str:
    output_value = record.get("output")
    if isinstance(output_value, str) and output_value.strip():
        parsed_tool = _extract_api_name_from_output(output_value)
        if parsed_tool:
            return _map_tool_name(parsed_tool)

    for key in ["expected_primary_tool", "tool_name", "tool", "api", "api_name", "gold_tool"]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            mapped = _map_tool_name(value)
            if mapped:
                return mapped

    query_like = " ".join(
        str(record.get(key, ""))
        for key in ["query", "instruction", "question", "task", "prompt"]
    ).lower()
    return _infer_tool_from_text(query_like)


def _extract_expected_substring(record: dict[str, Any]) -> str:
    output_value = record.get("output")
    if isinstance(output_value, str) and output_value.strip():
        api_name = _extract_api_name_from_output(output_value)
        if api_name:
            return api_name

    for key in ["expected_substring", "gold_answer", "answer", "expected_output", "reference"]:
        value = record.get(key)
        if isinstance(value, str):
            return value.strip()
    return ""


def _map_tool_name(raw: str) -> str:
    lowered = raw.strip().lower()
    if lowered in ALLOWED_TOOLS:
        return lowered

    # API-Bank style API function names are treated as tool calls in web_search-like bucket.
    if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", raw.strip()):
        return "web_search"

    if any(token in lowered for token in ["calc", "math", "arithmetic"]):
        return "calculator"
    if any(token in lowered for token in ["search", "web", "browser", "api"]):
        return "web_search"
    if any(token in lowered for token in ["knowledge", "kb", "local", "retriev"]):
        return "knowledge_base"
    return "text"


def _infer_tool_from_text(text: str) -> str:
    if any(token in text for token in ["add", "sum", "plus", "minus", "multiply", "divide", "calculate"]):
        return "calculator"
    if any(token in text for token in ["search", "internet", "web", "lookup"]):
        return "web_search"
    if any(token in text for token in ["knowledge base", "local file", "local knowledge", "kb"]):
        return "knowledge_base"
    return "text"


def _extract_apibank_user_query(raw_input: str) -> str:
    lines = [line.strip() for line in raw_input.splitlines() if line.strip()]
    user_lines = [line for line in lines if line.lower().startswith("user:")]
    if user_lines:
        return user_lines[-1].split(":", 1)[1].strip()

    marker = "Generate API Request:"
    if marker in raw_input:
        before = raw_input.split(marker)[0]
        user_chunks = re.findall(r"User:\s*(.+)", before)
        if user_chunks:
            return user_chunks[-1].strip()

    return ""


def _extract_api_name_from_output(output_text: str) -> str:
    # Handles patterns like:
    # API-Request: [Get_All_Sessions()]
    # API-Request: [book_retreat(...)]
    match = re.search(r"API-Request:\s*\[\s*([A-Za-z_][A-Za-z0-9_]*)", output_text)
    if match:
        return match.group(1)

    # Fallback for malformed bracket output observed in some samples.
    match = re.search(r"API-Request:\s*([A-Za-z_][A-Za-z0-9_]*)", output_text)
    if match:
        return match.group(1)
    return ""
