from __future__ import annotations

import json
from pathlib import Path

from atlas_agent.datasets import convert_external_dataset


def test_convert_external_dataset_json(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "id": "a1",
                        "instruction": "Search the web for JSONL format",
                        "tool_name": "web_search",
                        "gold_answer": "json lines",
                    },
                    {
                        "id": "a2",
                        "question": "Add 3 and 4",
                        "expected_primary_tool": "calculator",
                        "expected_substring": "7",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    out_path = tmp_path / "cases.json"
    cases = convert_external_dataset(source_path=source, out_path=out_path, source_format="auto")
    assert out_path.exists()
    assert len(cases) == 2
    assert all("query" in case for case in cases)
    assert all("expected_primary_tool" in case for case in cases)


def test_convert_external_dataset_limit_sampling(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    lines = [
        json.dumps({"id": f"c{i}", "query": f"task {i}", "tool": "text"})
        for i in range(10)
    ]
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_path = tmp_path / "cases_sample.json"
    cases = convert_external_dataset(source_path=source, out_path=out_path, limit=3, seed=7)
    assert len(cases) == 3


def test_convert_apibank_extracts_user_query_and_api_name(tmp_path: Path) -> None:
    source = tmp_path / "apibank.json"
    source.write_text(
        json.dumps(
            [
                {
                    "instruction": "Generate API request",
                    "input": "User: hello\nUser: Can you list all sessions?\nGenerate API Request:",
                    "output": "API-Request: [Get_All_Sessions()]",
                }
            ]
        ),
        encoding="utf-8",
    )

    out_path = tmp_path / "cases.json"
    cases = convert_external_dataset(source_path=source, out_path=out_path, source_format="apibank")
    assert len(cases) == 1
    assert cases[0]["query"] == "Can you list all sessions?"
    assert cases[0]["expected_primary_tool"] == "web_search"
    assert cases[0]["expected_substring"] == "Get_All_Sessions"