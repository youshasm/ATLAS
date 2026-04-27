# ATLAS Agentic System

ATLAS is a runnable agentic tool-use prototype with a planner, caller, summarizer, shared episodic memory, verifier, adaptive coordinator, and evaluation harness.

Implemented architecture improvements include:

- adaptive coordination modes (`direct_call`, `single_plan`, `iterative_plan`)
- verification of tool output structure
- recovery and re-planning loops for failed executions
- configurable memory retrieval (`lexical`, `vector`, `hybrid`)
- optional Ollama-based embeddings with automatic fallback to local hash embeddings

## LLM choice

Recommended for this semester project: use `ollama` for local reproducibility and low API cost.

- `rules` (default): deterministic, no model required
- `ollama`: local model via `http://localhost:11434`
- `online`: OpenAI-compatible endpoint (set `OPENAI_API_KEY`)

## Run

```bash
python -m atlas_agent.cli run --task "Find the total of 12 and 30"
```

Run with Ollama:

```bash
python -m atlas_agent.cli run --task "remember planner role" --llm-backend ollama --llm-model llama3.1:8b

# Hybrid memory retrieval with Ollama embeddings
python -m atlas_agent.cli run --task "what codename did I tell you" --memory-backend hybrid --embedding-provider ollama --embedding-model nomic-embed-text
```

Run with online LLM (OpenAI-compatible endpoint):

```bash
set OPENAI_API_KEY=your_key_here
python -m atlas_agent.cli run --task "remember planner role" --llm-backend online --llm-model gpt-4o-mini --llm-api-base https://api.openai.com
```

## Evaluate

```bash
python -m atlas_agent.cli evaluate
```

Evaluate with Ollama:

```bash
python -m atlas_agent.cli evaluate --llm-backend ollama --llm-model llama3.1:8b

# Evaluate with vector-only memory retrieval
python -m atlas_agent.cli evaluate --memory-backend vector --embedding-provider hash
```

## Benchmark (Baselines + Ablations)

Generate per-case CSV and summary JSON for paper tables:

```bash
python -m atlas_agent.cli benchmark --backends rules --include-no-memory --csv-path data/benchmark_rows.csv --json-path data/benchmark_summary.json
```

Include verifier ablation as well:

```bash
python -m atlas_agent.cli benchmark --backends rules --include-no-memory --include-no-verifier --csv-path data/benchmark_rows.csv --json-path data/benchmark_summary.json
```

Compare local model backend (requires Ollama running):

```bash
python -m atlas_agent.cli benchmark --backends rules ollama --llm-model llama3.1:8b --include-no-memory

# Benchmark hybrid retrieval and verifier/memory ablations
python -m atlas_agent.cli benchmark --backends rules --memory-backend hybrid --memory-vector-weight 0.7 --embedding-provider hash --include-no-memory --include-no-verifier
```

Run benchmark with custom converted cases:

```bash
python -m atlas_agent.cli benchmark --backends rules --cases-path data/external_cases.json --include-no-memory --include-no-verifier
```

The benchmark exports:

- `data/benchmark_rows.csv`: per-case records
- `data/benchmark_summary.json`: aggregated metrics for each variant

## Generate paper-ready report

```bash
python -m atlas_agent.cli report --csv-path data/benchmark_rows.csv --json-path data/benchmark_summary.json --out-path data/results_report.md
```

This creates a markdown file with a summary table and failure analysis.

## Memory Retrieval Options

- `--memory-backend lexical`: token overlap retrieval
- `--memory-backend vector`: embedding similarity retrieval
- `--memory-backend hybrid`: weighted lexical + vector retrieval
- `--memory-vector-weight 0.0..1.0`: vector contribution in hybrid mode
- `--embedding-provider hash|ollama`: embedding source for vector/hybrid modes
- `--embedding-api-base`: Ollama endpoint for embeddings (default `http://localhost:11434`)
- `--embedding-model`: Ollama embedding model (default `nomic-embed-text`)

## External Dataset Workflow

1. Download a benchmark dataset file (`.json`/`.jsonl`) from the internet.
2. Convert it to ATLAS benchmark case format.
3. Run the same ablations (`full`, `no_memory`, `no_verifier`) on converted cases.

Convert external dataset:

```bash
python -m atlas_agent.cli prepare-cases --source-path path/to/dataset.json --out-path data/external_cases.json --source-format auto --limit 200 --seed 42
```

Benchmark converted cases:

```bash
python -m atlas_agent.cli benchmark --backends rules ollama --cases-path data/external_cases.json --include-no-memory --include-no-verifier --csv-path data/external_rows.csv --json-path data/external_summary.json
python -m atlas_agent.cli report --csv-path data/external_rows.csv --json-path data/external_summary.json --out-path data/external_report.md
```

Supported `--source-format` values:

- `auto`: generic field detection
- `generic`: same as auto but without dataset-specific assumptions
- `apibank`: prefers API-Bank style keys (`instruction`, `question`, `tool_name`)
- `toolbench`: prefers ToolBench-like key patterns
