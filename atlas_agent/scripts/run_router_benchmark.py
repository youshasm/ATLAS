#!/usr/bin/env python3
"""Run the benchmark programmatically using the package run_benchmark function.
This avoids needing to install the package into the environment.
"""
from pathlib import Path
import json
import sys

repo_root = Path(__file__).resolve().parents[2]
# Ensure the package directory (atlas_agent/) is on sys.path
sys.path.insert(0, str(repo_root / 'atlas_agent'))

from atlas_agent.benchmark import run_benchmark


def main():
    repo_root = Path(__file__).resolve().parents[2]
    cases_path = repo_root / 'atlas_agent' / 'data' / 'apibank_lv1_cases_50.json'
    csv_path = repo_root / 'atlas_agent' / 'data' / 'apibank_router_rows.csv'
    json_path = repo_root / 'atlas_agent' / 'data' / 'apibank_router_summary.json'

    output = run_benchmark(
        backends=['rules'],
        llm_model='rules',
        llm_api_base='http://localhost:11434',
        memory_backend='hybrid',
        memory_vector_weight=0.7,
        embedding_provider='hash',
        embedding_api_base='http://localhost:11434',
        embedding_model='nomic-embed-text',
        include_no_memory=True,
        include_no_verifier=True,
        cases=[c for c in json.loads(cases_path.read_text(encoding='utf-8'))],
        csv_path=csv_path,
        json_path=json_path,
    )

    print(json.dumps(output['summaries'], indent=2))


if __name__ == '__main__':
    main()
