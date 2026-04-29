#!/usr/bin/env python3
"""Train a lightweight tool-router classifier on converted API-Bank cases.

Saves model to atlas_agent/models/router.pkl
"""
import json
import os
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def load_cases(path):
    with open(path, 'r', encoding='utf-8') as f:
        cases = json.load(f)
    # cases expected as list of dicts
    return cases


def prepare_xy(cases):
    X = [c.get('query','') for c in cases]
    y = [c.get('expected_primary_tool','text') for c in cases]
    return X, y


def main():
    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / 'atlas_agent' / 'data' / 'apibank_lv1_cases_50.json'
    if not data_path.exists():
        data_path = repo_root / 'atlas_agent' / 'data' / 'apibank_lv1_cases.json'
    if not data_path.exists():
        raise FileNotFoundError(f"No apibank cases found at {data_path}")

    cases = load_cases(data_path)
    X, y = prepare_xy(cases)

    clf = make_pipeline(TfidfVectorizer(ngram_range=(1,2), max_features=4096), LogisticRegression(max_iter=1000))
    clf.fit(X, y)

    models_dir = repo_root / 'atlas_agent' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / 'router.pkl'
    joblib.dump(clf, out_path)
    print(f"Saved router model to {out_path}")


if __name__ == '__main__':
    main()
