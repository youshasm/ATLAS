# Diagram Prompts for the ATLAS Paper

Use these prompts in a diagram generator or hand them to a designer. They are written to match the current paper draft and the ATLAS architecture.

## Figure 1: ATLAS System Architecture

Create a clean IEEE-style research diagram of the ATLAS multi-LLM agent architecture. The diagram should be landscape-oriented, minimal, and publication-ready. Use a white background, thin black or dark-gray lines, and a restrained accent color such as teal or blue. Show the following modules as labeled boxes connected by arrows:

- User Query
- Adaptive Coordinator
- Planner
- Shared Episodic Memory
- Caller / Tool Executor
- Verifier
- Recovery Policy
- Summarizer
- Final Answer

The flow should be:

1. User Query enters Adaptive Coordinator.
2. Adaptive Coordinator routes to Planner.
3. Planner consults Shared Episodic Memory.
4. Planner sends actions to Caller / Tool Executor.
5. Tool results go to Verifier.
6. If verification fails, Recovery Policy triggers replanning and retries.
7. If verification succeeds, Summarizer produces the Final Answer.
8. Memory should receive successful traces and useful failure traces.

Add small annotations for the key design ideas:
- memory-augmented planning
- verification-guided recovery
- adaptive routing for cost control

Keep text concise and readable at journal-column scale.

## Figure 2: Benchmark and Evaluation Flow

Create a second IEEE-style diagram showing the evaluation pipeline for ATLAS. Use the same visual style as Figure 1. The diagram should show:

- Dataset Source, with examples such as internal benchmark, API-Bank, and other external cases
- Case Conversion / Normalization
- Benchmark Runner
- ATLAS Variants: Full, No Memory, No Verifier
- Metrics Output: verified rate, case pass rate, average steps, iterative mode rate
- Report Generation

The flow should be:

1. Dataset Source feeds Case Conversion / Normalization.
2. Converted cases go into the Benchmark Runner.
3. Benchmark Runner evaluates the Full, No Memory, and No Verifier variants.
4. Metrics are collected and written to a Report Generation block.
5. Report Generation outputs tables and ablation summaries used in the paper.

Add a short callout noting that API-Bank was downloaded from Hugging Face and converted into ATLAS case format for external evaluation.

## Optional Figure 3: Ablation Summary Graphic

Create a compact bar-chart style academic figure comparing the three benchmark variants across the main metrics. Use a simple and publication-friendly aesthetic. Plot:

- Full ATLAS
- No Memory
- No Verifier

Include the following metrics:
- verified rate
- case pass rate
- average steps

Keep the labels minimal and readable. Use a legend and avoid decorative effects.

## Short Version for Designers

If you need a very short prompt, use this:

Design two publication-ready IEEE figures for a paper on ATLAS, a memory-augmented multi-LLM agent. Figure 1 should show the architecture flow from query to planner, memory, tool execution, verifier, recovery, summarizer, and final answer. Figure 2 should show the benchmark pipeline from dataset conversion to benchmark runner to ablation variants and report generation. Minimal, professional, landscape layout, white background, dark lines, one subtle accent color, suitable for a conference paper.
