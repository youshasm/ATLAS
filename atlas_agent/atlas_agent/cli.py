from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import load_cases, run_benchmark
from .config import default_config
from .core import AtlasAgent
from .datasets import convert_external_dataset
from .evaluation import evaluate_agent
from .reporting import generate_markdown_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlas-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the ATLAS agent on a single query")
    run_parser.add_argument("--task", required=True, help="User query or task")
    run_parser.add_argument("--llm-backend", choices=["rules", "ollama", "online"], default="rules")
    run_parser.add_argument("--llm-model", default="llama3.1:8b")
    run_parser.add_argument("--llm-api-base", default="http://localhost:11434")
    run_parser.add_argument("--memory-backend", choices=["lexical", "vector", "hybrid"], default="hybrid")
    run_parser.add_argument("--memory-vector-weight", type=float, default=0.7)
    run_parser.add_argument("--embedding-provider", choices=["hash", "ollama"], default="hash")
    run_parser.add_argument("--embedding-api-base", default="http://localhost:11434")
    run_parser.add_argument("--embedding-model", default="nomic-embed-text")

    eval_parser = subparsers.add_parser("evaluate", help="Run the default evaluation set")
    eval_parser.add_argument("--llm-backend", choices=["rules", "ollama", "online"], default="rules")
    eval_parser.add_argument("--llm-model", default="llama3.1:8b")
    eval_parser.add_argument("--llm-api-base", default="http://localhost:11434")
    eval_parser.add_argument("--memory-backend", choices=["lexical", "vector", "hybrid"], default="hybrid")
    eval_parser.add_argument("--memory-vector-weight", type=float, default=0.7)
    eval_parser.add_argument("--embedding-provider", choices=["hash", "ollama"], default="hash")
    eval_parser.add_argument("--embedding-api-base", default="http://localhost:11434")
    eval_parser.add_argument("--embedding-model", default="nomic-embed-text")

    benchmark_parser = subparsers.add_parser("benchmark", help="Run baseline/ablation benchmark and export CSV/JSON")
    benchmark_parser.add_argument("--backends", nargs="+", choices=["rules", "ollama", "online"], default=["rules"])
    benchmark_parser.add_argument("--llm-model", default="llama3.1:8b")
    benchmark_parser.add_argument("--llm-api-base", default="http://localhost:11434")
    benchmark_parser.add_argument("--memory-backend", choices=["lexical", "vector", "hybrid"], default="hybrid")
    benchmark_parser.add_argument("--memory-vector-weight", type=float, default=0.7)
    benchmark_parser.add_argument("--embedding-provider", choices=["hash", "ollama"], default="hash")
    benchmark_parser.add_argument("--embedding-api-base", default="http://localhost:11434")
    benchmark_parser.add_argument("--embedding-model", default="nomic-embed-text")
    benchmark_parser.add_argument("--include-no-memory", action="store_true")
    benchmark_parser.add_argument("--include-no-verifier", action="store_true")
    benchmark_parser.add_argument("--include-react-baseline", action="store_true")
    benchmark_parser.add_argument("--include-decomposition-baseline", action="store_true")
    benchmark_parser.add_argument("--cases-path", default="", help="Optional path to custom benchmark cases (.json/.jsonl)")
    benchmark_parser.add_argument("--csv-path", default="data/benchmark_rows.csv")
    benchmark_parser.add_argument("--json-path", default="data/benchmark_summary.json")

    prepare_parser = subparsers.add_parser("prepare-cases", help="Convert an external dataset file into ATLAS benchmark cases")
    prepare_parser.add_argument("--source-path", required=True, help="Input dataset path (.json/.jsonl/.ndjson)")
    prepare_parser.add_argument("--out-path", default="data/external_cases.json", help="Output ATLAS cases file")
    prepare_parser.add_argument("--source-format", choices=["auto", "generic", "apibank", "toolbench"], default="auto")
    prepare_parser.add_argument("--limit", type=int, default=0, help="Optional sample size (0 means all records)")
    prepare_parser.add_argument("--seed", type=int, default=42, help="Sampling seed when --limit is used")

    report_parser = subparsers.add_parser("report", help="Generate paper-ready markdown report from benchmark outputs")
    report_parser.add_argument("--csv-path", default="data/benchmark_rows.csv")
    report_parser.add_argument("--json-path", default="data/benchmark_summary.json")
    report_parser.add_argument("--out-path", default="data/results_report.md")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        config = default_config(
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
            llm_api_base=args.llm_api_base,
            memory_backend=args.memory_backend,
            memory_vector_weight=args.memory_vector_weight,
            embedding_provider=args.embedding_provider,
            embedding_api_base=args.embedding_api_base,
            embedding_model=args.embedding_model,
        )
        agent = AtlasAgent(config)
        result = agent.run(args.task)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "evaluate":
        config = default_config(
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
            llm_api_base=args.llm_api_base,
            memory_backend=args.memory_backend,
            memory_vector_weight=args.memory_vector_weight,
            embedding_provider=args.embedding_provider,
            embedding_api_base=args.embedding_api_base,
            embedding_model=args.embedding_model,
        )
        result = evaluate_agent(config)
        print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))
        return

    if args.command == "benchmark":
        cases = load_cases(Path(args.cases_path)) if args.cases_path else None
        output = run_benchmark(
            backends=args.backends,
            llm_model=args.llm_model,
            llm_api_base=args.llm_api_base,
            memory_backend=args.memory_backend,
            memory_vector_weight=args.memory_vector_weight,
            embedding_provider=args.embedding_provider,
            embedding_api_base=args.embedding_api_base,
            embedding_model=args.embedding_model,
            include_no_memory=args.include_no_memory,
            include_no_verifier=args.include_no_verifier,
            include_react_baseline=args.include_react_baseline,
            include_decomposition_baseline=args.include_decomposition_baseline,
            cases=cases,
            csv_path=Path(args.csv_path),
            json_path=Path(args.json_path),
        )
        print(json.dumps(output["summaries"], indent=2, ensure_ascii=False))
        return

    if args.command == "report":
        report_path = generate_markdown_report(
            csv_path=Path(args.csv_path),
            json_path=Path(args.json_path),
            out_path=Path(args.out_path),
        )
        print(json.dumps({"report": str(report_path)}, indent=2, ensure_ascii=False))
        return

    if args.command == "prepare-cases":
        cases = convert_external_dataset(
            source_path=Path(args.source_path),
            out_path=Path(args.out_path),
            source_format=args.source_format,
            limit=args.limit,
            seed=args.seed,
        )
        print(json.dumps({"out_path": args.out_path, "cases": len(cases)}, indent=2, ensure_ascii=False))
        return

    raise SystemExit(1)


if __name__ == "__main__":
    main()
