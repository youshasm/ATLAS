from __future__ import annotations

import json
from pathlib import Path

from atlas_agent.benchmark import load_cases, run_benchmark


def test_benchmark_exports(tmp_path: Path) -> None:
    csv_path = tmp_path / "rows.csv"
    json_path = tmp_path / "summary.json"

    payload = run_benchmark(
        backends=["rules"],
        llm_model="llama3.1:8b",
        llm_api_base="http://localhost:11434",
        include_no_memory=True,
        csv_path=csv_path,
        json_path=json_path,
    )

    assert csv_path.exists()
    assert json_path.exists()
    assert "summaries" in payload
    assert len(payload["summaries"]) >= 1

    first = payload["summaries"][0]
    assert "case_pass_rate" in first
    assert "expected_substring_match_rate" in first

    first_row = payload["rows"][0]
    assert "coordination_mode" in first_row
    assert "retries" in first_row
    assert "memory_backend" in first_row
    assert "embedding_provider" in first_row


def test_load_cases_json(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "external_1",
                        "query": "Find the total of 2 and 3",
                        "expected_primary_tool": "calculator",
                        "expected_substring": "5",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    cases = load_cases(cases_path)
    assert len(cases) == 1
    assert cases[0]["id"] == "external_1"
    assert cases[0]["expected_verified"] is True
