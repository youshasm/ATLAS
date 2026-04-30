from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


def choose_step_with_llm(
    backend: str,
    model: str,
    api_base: str,
    api_key_env: str,
    query: str,
    memory_context: list[dict[str, Any]],
) -> dict[str, Any] | None:
    prompt = build_prompt(query, memory_context)

    try:
        if backend == "ollama":
            return _call_ollama(model=model, api_base=api_base, prompt=prompt)
        if backend == "online":
            return _call_openai_compatible(model=model, api_base=api_base, api_key_env=api_key_env, prompt=prompt)
        return None
    except Exception:
        return None


def build_prompt(query: str, memory_context: list[dict[str, Any]]) -> str:
    memory_lines = []
    for item in memory_context[:3]:
        tool_seq = ",".join(item.get("tool_sequence", []))
        memory_lines.append(f"- query={item.get('query','')} tools={tool_seq} success={item.get('success', True)}")
    memory_text = "\n".join(memory_lines) if memory_lines else "- none"

    return (
        "You are a planner for a tool-using agent.\n"
        "Available tools: calculator, knowledge_base, web_search, text.\n"
        "Return ONLY valid compact JSON with keys: tool_name, arguments, purpose, function_name.\n"
        "tool_name must be one of calculator|knowledge_base|web_search|text.\n"
        "CRITICAL: If the user explicitly names a tool (calculator/knowledge_base/web_search/text), you MUST choose that tool.\n"
        "CRITICAL: Do NOT let memory context override an explicit user instruction.\n"
        "If arithmetic is requested, choose calculator and include arguments.expression.\n"
        "If the user asks to search/find/lookup information, choose web_search and include arguments.query.\n"
        "Only choose knowledge_base for EXPLICIT LOCAL lookups (mentions 'knowledge base', 'local knowledge base', 'local file', 'atlas kb', or similar) and include arguments.query.\n"
        "If the user says to store/save a note or remember a note, choose text and include arguments.message (do NOT use knowledge_base).\n"
        "Otherwise choose text and include arguments.message.\n"
        "For function_name: Extract the most likely specific API function name from the user's query.\n"
        "Examples: 'list reminders' -> List_Reminders, 'get records' -> get_records, 'make payment' -> make_payment.\n"
        "If unclear, use the action verb from the query (e.g., search, get, list, update, create, delete).\n"
        "If no clear function, leave function_name empty.\n"
        "Examples:\n"
        "- 'Use the web_search tool to search the web for JSONL' -> web_search\n"
        "- 'Search the web for JSONL' -> web_search\n"
        "- 'Find planner and summarizer roles in the atlas knowledge base' -> knowledge_base\n"
        "- 'Store this note: codename is X' -> text\n"
        f"User query: {query}\n"
        f"Memory context (optional):\n{memory_text}\n"
    )


def _call_ollama(model: str, api_base: str, prompt: str) -> dict[str, Any] | None:
    url = api_base.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    raw = _post_json(url, payload, headers={})
    response_text = raw.get("response", "")
    return _parse_json_payload(response_text)


def _call_openai_compatible(model: str, api_base: str, api_key_env: str, prompt: str) -> dict[str, Any] | None:
    api_key = os.getenv(api_key_env, "")
    if not api_key:
        return None

    base = api_base.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    url = base + "/chat/completions"

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "Return only JSON."},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    raw = _post_json(url, payload, headers=headers)
    choices = raw.get("choices", [])
    if not choices:
        return None
    content = choices[0].get("message", {}).get("content", "")
    return _parse_json_payload(content)


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read().decode("utf-8")
    return json.loads(data)


def _parse_json_payload(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict):
        return None
    tool_name = data.get("tool_name")
    if tool_name not in {"calculator", "knowledge_base", "web_search", "text"}:
        return None
    args = data.get("arguments", {})
    if not isinstance(args, dict):
        return None
    purpose = str(data.get("purpose", ""))
    return {"tool_name": tool_name, "arguments": args, "purpose": purpose}
