from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def generate_markdown_report(csv_path: Path, json_path: Path, out_path: Path) -> Path:
    summaries, rows = _load_data(csv_path, json_path)

    lines: list[str] = []
    lines.append("# ATLAS Benchmark Report")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append(
        "| Variant | Backend | Memory Backend | Embedding | Memory | Verifier | Cases | Verified Rate | Tool Match Rate | Substring Match Rate | Case Pass Rate | Avg Steps | Avg Tool Calls | Avg Retries | Iterative Rate |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in summaries:
        lines.append(
            f"| {item.get('variant')} | {item.get('backend')} | {item.get('memory_backend')} | {item.get('embedding_provider')} | "
            f"{int(bool(item.get('use_memory')))} | {int(bool(item.get('use_verifier')))} | "
            f"{item.get('cases')} | {float(item.get('success_rate', 0.0)):.2f} | {float(item.get('primary_tool_match_rate', 0.0)):.2f} | "
            f"{float(item.get('expected_substring_match_rate', 0.0)):.2f} | {float(item.get('case_pass_rate', 0.0)):.2f} | "
            f"{float(item.get('avg_steps', 0.0)):.2f} | {float(item.get('avg_tool_calls', 0.0)):.2f} | "
            f"{float(item.get('avg_retries', 0.0)):.2f} | {float(item.get('iterative_mode_rate', 0.0)):.2f} |"
        )

    lines.append("")
    lines.append("## Failure Analysis")
    lines.append("")
    failures = [row for row in rows if str(row.get("case_pass", "")).lower() in {"false", "0"}]
    lines.append(f"- Total cases: {len(rows)}")
    lines.append(f"- Failed cases: {len(failures)}")

    if failures:
        variant_counts = Counter(f"{f.get('backend')}::{f.get('variant')}" for f in failures)
        lines.append("- Failure counts by variant:")
        for key, count in sorted(variant_counts.items()):
            lines.append(f"  - {key}: {count}")

        lines.append("")
        lines.append("### Failed Cases")
        lines.append("")
        lines.append("| Case ID | Variant | Backend | Mode | Retries | Expected Tool | Predicted Tool | Verified | Expected Verified | Expected Substring |")
        lines.append("|---|---|---|---|---:|---|---|---:|---:|---|")
        for f in failures:
            lines.append(
                f"| {f.get('case_id')} | {f.get('variant')} | {f.get('backend')} | "
                f"{f.get('coordination_mode')} | {f.get('retries')} | "
                f"{f.get('expected_primary_tool')} | {f.get('predicted_primary_tool')} | "
                f"{f.get('verified')} | {f.get('expected_verified')} | {f.get('expected_substring')} |"
            )
    else:
        lines.append("- No failed cases in this run.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _load_data(csv_path: Path, json_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        summaries = payload.get("summaries", [])

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

    return summaries, rows
