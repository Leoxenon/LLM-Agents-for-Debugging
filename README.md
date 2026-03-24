# LLM Debugging Evaluation Framework

This project is a minimal, reproducible experimental framework for comparing two LLM-based debugging setups:

- `baseline`: single-shot code fixing with no execution feedback
- `agent`: LangChain-based iterative debugging with Python self-check execution feedback

The framework is designed for analysis, not performance. It saves full traces, per-case results, and aggregate metrics for report writing and failure analysis.

## Project Structure

- `main.py`: experiment entrypoint
- `llm.py`: unified LLM wrapper using `API_KEY`, `BASE_URL`, and `MODEL_NAME`
- `agent.py`: ReAct-style iterative debugging agent with execution tool
- `executor.py`: isolated Python code execution
- `dataset.py`: dataset loader
- `evaluator.py`: baseline and agent evaluation logic
- `utils.py`: output helpers, code extraction, and failure classification
- `dataset.json`: fixed 12-case benchmark
- `requirements.txt`: dependencies

## Requirements

- Python 3.10+
- An OpenAI-compatible chat completion endpoint

The wrapper is intentionally provider-flexible through environment variables, so it can be used with providers that expose an OpenAI-compatible API surface, including custom `BASE_URL` values.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Set environment variables:

```bash
export API_KEY="your_api_key"
export BASE_URL="https://your-provider-endpoint/v1"
export MODEL_NAME="your-model-name"
```

On PowerShell:

```powershell
$env:API_KEY="your_api_key"
$env:BASE_URL="https://your-provider-endpoint/v1"
$env:MODEL_NAME="your-model-name"
```

## Run

```bash
python main.py
```

The program writes:

- `full_traces.json`
- `results.csv`
- `metrics.json`

It also prints:

- baseline success rate
- agent success rate
- improvement

## Trace Format

Each record in `full_traces.json` follows this structure:

```json
{
  "case_id": "case_1",
  "setup": "agent",
  "iterations": [
    {
      "step": 1,
      "llm_output": "...",
      "execution_output": "...",
      "error": "...",
      "success": false
    }
  ],
  "final_success": true
}
```

Agent traces also include `reasoning_trace` fields and a top-level `agent_reasoning_traces` list containing raw ReAct-style outputs and tool observations.

The official benchmark `test_code` is only used during final evaluation. The agent may run its own self-check scripts during iteration, but it does not directly receive or execute the benchmark test cases inside the agent loop.

## Failure Types

The framework classifies failed runs using simple heuristics:

- `error_misinterpretation`
- `wrong_fix`
- `runtime_error`
- `loop_failure`

These labels are intended to support qualitative analysis in the report rather than serve as a perfect taxonomy.
