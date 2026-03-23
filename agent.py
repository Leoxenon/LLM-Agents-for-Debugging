import io
from contextlib import redirect_stdout
from typing import Any, Dict, List

from langchain_core.tools import tool

from executor import render_execution_feedback, run_python_code
from llm import UnifiedLLM
from utils import extract_python_code


AGENT_SYSTEM_PROMPT = """You are a ReAct-style Python debugging agent.
You may inspect execution feedback to improve a candidate fix.
At each step:
1. Think briefly about the bug.
2. Produce a complete corrected Python program.
3. Use the execution feedback from previous attempts when available.

Return your answer using this exact structure:
Thought: <short reasoning>
Action: propose_fix
Action Input:
```python
<complete corrected code>
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
            "Produce a complete corrected Python program."
        )

    def _build_retry_prompt(self, case: Dict[str, str], previous_code: str, execution_feedback: str) -> str:
        return (
            f"Task:\n{case['task']}\n\n"
            f"Original buggy code:\n```python\n{case['buggy_code']}\n```\n\n"
            f"Previous candidate code:\n```python\n{previous_code}\n```\n\n"
            f"Execution feedback:\n{execution_feedback}\n\n"
            "Revise the fix. Return a complete corrected Python program in the required format."
        )

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
            candidate_code = extract_python_code(raw_output)
            execution_input = candidate_code + "\n" + case["test_code"]

            captured_stdout = io.StringIO()
            with redirect_stdout(captured_stdout):
                tool_feedback = python_execution_tool.invoke({"code_with_test": execution_input})

            execution = run_python_code(execution_input)
            execution_feedback = render_execution_feedback(execution)

            reasoning_trace = {
                "raw_agent_output": raw_output,
                "tool_trace": tool_feedback,
                "verbose_log": captured_stdout.getvalue().strip(),
            }
            if self.verbose:
                self.trace_collector.record(raw_output)
                self.trace_collector.record(tool_feedback)

            iteration = {
                "step": step,
                "llm_output": raw_output,
                "execution_output": execution["stdout"],
                "error": execution["stderr"],
                "success": execution["success"],
                "reasoning_trace": reasoning_trace,
            }
            iterations.append(iteration)

            if execution["success"]:
                return {
                    "iterations": iterations,
                    "final_success": True,
                    "final_code": candidate_code,
                    "agent_reasoning_traces": self.trace_collector.events,
                }

            previous_code = candidate_code

        return {
            "iterations": iterations,
            "final_success": False,
            "final_code": previous_code,
            "agent_reasoning_traces": self.trace_collector.events,
        }
