import io
from contextlib import redirect_stdout
import re
from typing import Any, Dict, List

from langchain_core.tools import tool

from executor import render_execution_feedback, run_python_code
from llm import UnifiedLLM
from utils import extract_python_code


AGENT_SYSTEM_PROMPT = """You are a ReAct-style Python debugging agent.
You may inspect execution feedback from your own self-checks to improve a candidate fix.
At each step:
1. Think briefly about the bug.
2. Decide whether to run a self-check or submit the current fix.
3. If you run a self-check, provide the exact Python script to execute.
4. Use the execution feedback from previous attempts when available.

Return your answer using this exact structure:
Thought: <short reasoning>
Action: run_check OR submit
Check Input:
```python
<complete executable Python script for your self-check, including the candidate program and any checks you want to run>
```
Final Code:
```python
<complete corrected Python program only>
```
"""


@tool
def python_execution_tool(code_with_test: str) -> str:
    """Execute Python code and return structured stdout and stderr."""
    result = run_python_code(code_with_test)
    return render_execution_feedback(result)


class AgentTraceCollector:
    def __init__(self) -> None:
        self.events: List[str] = []

    def record(self, text: str) -> None:
        if text.strip():
            self.events.append(text.strip())


class DebugAgent:
    def __init__(self, llm: UnifiedLLM, max_iterations: int = 4, verbose: bool = True):
        self.llm = llm
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.trace_collector = AgentTraceCollector()

    def _build_initial_prompt(self, case: Dict[str, str]) -> str:
        return (
            f"Task:\n{case['task']}\n\n"
            f"Buggy code:\n```python\n{case['buggy_code']}\n```\n\n"
            "You may use self-check execution, but you must not rely on hidden benchmark tests. "
            "Decide whether to run a self-check or submit a final answer."
        )

    def _build_retry_prompt(self, case: Dict[str, str], previous_code: str, execution_feedback: str) -> str:
        return (
            f"Task:\n{case['task']}\n\n"
            f"Original buggy code:\n```python\n{case['buggy_code']}\n```\n\n"
            f"Previous candidate code:\n```python\n{previous_code}\n```\n\n"
            f"Self-check execution feedback:\n{execution_feedback}\n\n"
            "Revise the fix. Decide whether to run another self-check or submit the final code."
        )

    def _extract_named_block(self, text: str, header: str) -> str:
        pattern = rf"{re.escape(header)}\s*```python\s*(.*?)```"
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip() + "\n"
        return ""

    def _parse_agent_output(self, raw_output: str) -> Dict[str, str]:
        action_match = re.search(r"Action:\s*(.+)", raw_output, flags=re.IGNORECASE)
        action = action_match.group(1).strip().lower() if action_match else "submit"
        if action not in {"run_check", "submit"}:
            action = "submit"

        final_code = self._extract_named_block(raw_output, "Final Code:")
        if not final_code:
            final_code = extract_python_code(raw_output)

        check_input = self._extract_named_block(raw_output, "Check Input:")
        return {
            "action": action,
            "final_code": final_code,
            "check_input": check_input,
        }

    def run_case(self, case: Dict[str, str]) -> Dict[str, Any]:
        self.trace_collector = AgentTraceCollector()
        iterations: List[Dict[str, Any]] = []
        previous_code = ""
        execution_feedback = ""

        for step in range(1, self.max_iterations + 1):
            prompt = (
                self._build_initial_prompt(case)
                if step == 1
                else self._build_retry_prompt(case, previous_code, execution_feedback)
            )

            raw_output = self.llm.invoke(prompt, system_prompt=AGENT_SYSTEM_PROMPT)
            parsed_output = self._parse_agent_output(raw_output)
            candidate_code = parsed_output["final_code"]
            action = parsed_output["action"]
            check_input = parsed_output["check_input"]

            tool_feedback = ""
            execution_output = ""
            error = ""
            step_success = None
            captured_stdout = io.StringIO()

            if action == "run_check":
                if check_input.strip():
                    with redirect_stdout(captured_stdout):
                        tool_feedback = python_execution_tool.invoke({"code_with_test": check_input})

                    execution = run_python_code(check_input)
                    execution_feedback = render_execution_feedback(execution)
                    execution_output = execution["stdout"]
                    error = execution["stderr"]
                    step_success = execution["success"]
                else:
                    execution_feedback = "No self-check script was provided."
                    error = execution_feedback
                    step_success = False
            else:
                execution_feedback = "Agent chose to submit the current candidate without another self-check."

            reasoning_trace = {
                "raw_agent_output": raw_output,
                "action": action,
                "self_check_code": check_input,
                "tool_trace": tool_feedback,
                "verbose_log": captured_stdout.getvalue().strip(),
            }
            if self.verbose:
                self.trace_collector.record(raw_output)
                if tool_feedback:
                    self.trace_collector.record(tool_feedback)

            iteration = {
                "step": step,
                "llm_output": raw_output,
                "candidate_code": candidate_code,
                "execution_output": execution_output,
                "error": error,
                "success": step_success,
                "reasoning_trace": reasoning_trace,
            }
            iterations.append(iteration)

            if action == "submit":
                return {
                    "iterations": iterations,
                    "final_code": candidate_code,
                    "agent_reasoning_traces": self.trace_collector.events,
                }

            previous_code = candidate_code

        return {
            "iterations": iterations,
            "final_code": previous_code,
            "agent_reasoning_traces": self.trace_collector.events,
        }
