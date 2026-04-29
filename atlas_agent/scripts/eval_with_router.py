#!/usr/bin/env python3
"""Evaluate API-Bank subset using the trained router and compare metrics.

Loads atlas_agent/models/router.pkl and atlas_agent/data/apibank_lv1_cases_50.json
Writes updated summary to atlas_agent/data/apibank_lv1_cases_50_with_router_summary.json
"""
import json
from pathlib import Path
import joblib
from collections import Counter


def load_cases(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate(cases, model):
    X = [c['query'] for c in cases]
    preds = model.predict(X)
    rows = []
    for c, p in zip(cases, preds):
        row = dict(c)
        row['router_predicted_primary_tool'] = p
        row['router_primary_tool_match'] = (p == c.get('expected_primary_tool'))
        row['router_case_pass'] = row['router_primary_tool_match'] and bool(c.get('expected_verified', True))
        rows.append(row)
    summary = {
        'cases': len(rows),
        'router_primary_tool_match_rate': sum(r['router_primary_tool_match'] for r in rows)/len(rows),
        'router_case_pass_rate': sum(r['router_case_pass'] for r in rows)/len(rows)
    }
    return summary, rows


def main():
    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / 'atlas_agent' / 'data' / 'apibank_lv1_cases_50.json'
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    model_path = repo_root / 'atlas_agent' / 'models' / 'router.pkl'
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    cases = load_cases(data_path)
    model = joblib.load(model_path)
    summary, rows = evaluate(cases, model)

    out_summary = repo_root / 'atlas_agent' / 'data' / 'apibank_lv1_cases_50_with_router_summary.json'
    out_rows = repo_root / 'atlas_agent' / 'data' / 'apibank_lv1_cases_50_with_router_rows.json'
    with open(out_summary, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    with open(out_rows, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)

    print('Wrote', out_summary, out_rows)


if __name__ == '__main__':
    main()
