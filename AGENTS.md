# AGENTS.md

## 1. Project Overview

This project is an academic experimental framework for evaluating LLM-based debugging systems.

We compare two setups:

1. Baseline LLM (single-shot code generation, no execution feedback)
2. Agent system (LangChain-based, with execution + iterative refinement)

The goal is NOT to build a powerful coding agent.

The goal is:

* to observe behavior
* to collect reasoning traces
* to identify limitations and failure patterns

---

## 2. Core Philosophy

### 2.1 Analysis > Performance

A system that fails in interesting ways is more valuable than one that always succeeds.

Focus on:

* why the model fails
* how it fails
* whether it improves with feedback

---

### 2.2 Simplicity > Complexity

DO:

* keep code minimal
* use simple abstractions
* make everything readable

DO NOT:

* build complex frameworks
* introduce unnecessary dependencies
* over-engineer

---

### 2.3 Reproducibility

All experiments must:

* run deterministically (as much as possible)
* produce consistent outputs
* save all results to files

---

## 3. System Architecture

### 3.1 Components

The system consists of:

* LLM Wrapper (llm.py)
* Baseline Generator
* Agent (LangChain-based)
* Code Executor
* Dataset Loader
* Evaluator
* Metrics Analyzer

---

### 3.2 Baseline

* Single LLM call
* No execution
* No iteration

Input:

* buggy_code
* task

Output:

* fixed_code

---

### 3.3 Agent

Built using LangChain

Capabilities:

* tool use (Python execution)
* iterative refinement (3–5 steps max)

Loop:

1. Generate code
2. Execute code
3. Observe output/error
4. Update prompt
5. Retry

---

## 4. Trace Logging (CRITICAL)

The system MUST log detailed traces for analysis.

For each test case and each setup:

* LLM outputs
* execution results
* errors
* iteration history

Agent-specific:

* reasoning traces (Thought / Action / Observation if available)

---

### 4.1 Trace Format

Example:

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

---

### 4.2 Storage

All traces must be saved to:

* full_traces.json

These traces are REQUIRED for report analysis.

---

## 5. Dataset Design

### 5.1 Dataset Philosophy

The dataset is intentionally designed to:

* expose weaknesses of LLMs
* trigger failure modes
* generate meaningful traces

Avoid trivial problems.

---

### 5.2 Dataset Size

* 12 test cases (fixed)

---

### 5.3 Bug Categories

Each test case belongs to one of:

1. Hidden logic bug
2. Misleading error
3. Multi-step reasoning bug
4. State-related bug
5. Bug causing repeated failure

---

### 5.4 Dataset (Fixed)

DO NOT modify unless necessary.

Each test case includes:

* buggy_code
* test_code

Evaluation is done by executing:

```
generated_code + test_code
```

---

## 6. Evaluation Protocol

### 6.1 Unified Evaluation

Both Baseline and Agent are evaluated the same way:

1. Generate code
2. Append test_code
3. Execute
4. Record result

---

### 6.2 Success Definition

Success = all assertions pass without error

---

### 6.3 Fair Comparison

* Baseline: no execution feedback
* Agent: has execution + iteration

---

## 7. Metrics

### 7.1 Core Metrics

1. Success Rate
2. Average Iterations (agent only)

---

### 7.2 Failure Classification

Failures should be categorized into:

* error_misinterpretation
* wrong_fix
* runtime_error
* loop_failure

---

### 7.3 Failure Analysis Goal

We aim to identify patterns such as:

* misunderstanding of error messages
* superficial fixes
* repeated incorrect strategies
* unstable reasoning across iterations

---

## 8. Expected Failure Modes

The system should expose:

### 8.1 Error Misinterpretation

LLM misunderstands the error message

---

### 8.2 Superficial Fix

Fix does not address root cause

---

### 8.3 Tool Misuse

Agent fails to properly use execution feedback

---

### 8.4 Infinite Loop (IMPORTANT)

Agent repeats same incorrect fix

---

### 8.5 Planning Collapse

Agent loses track of goal after multiple iterations

---

## 9. Code Design Guidelines

* keep functions short
* avoid deep nesting
* use clear naming
* add comments where necessary

---

## 10. What NOT to Do

DO NOT:

* build a full IDE agent
* add GUI
* use multi-agent systems
* implement memory systems
* optimize for performance

---

## 11. Outputs

The system must generate:

* full_traces.json (detailed logs)
* results.csv (summary per case)
* metrics.json (aggregated metrics)

---

## 12. Running the Project

The project must run with:

```
pip install -r requirements.txt
python main.py
```

User only configures:

* API_KEY
* BASE_URL
* MODEL_NAME

---

## 13. Extension Guidelines (Optional)

If extending the project:

Allowed:

* adding new test cases
* improving logging
* refining failure classification

Avoid:

* increasing complexity
* changing core evaluation logic

---

## 14. Final Reminder

This project is about understanding LLM limitations.

A system that produces rich, interpretable failures is more valuable than one that achieves high accuracy.

Focus on:

* trace quality
* failure diversity
* clarity of results
