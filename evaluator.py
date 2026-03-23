from collections import Counter
from typing import Any, Dict, List

from agent import DebugAgent
from dataset import DebugCase
from executor import run_python_code
from llm import SYSTEM_PROMPT, UnifiedLLM
from utils import classify_failure, extract_python_code, mean


class BaselineRunner:
    def __init__(self, llm: UnifiedLLM):
        self.llm = llm

    def run_case(self, case: DebugCase) -> Dict[str, Any]:
        prompt = self.llm.build_fix_prompt(case.buggy_code, case.task)
        raw_output = self.llm.invoke(prompt, system_prompt=SYSTEM_PROMPT)
        candidate_code = extract_python_code(raw_output)
        execution = run_python_code(candidate_code + "\n" + case.test_code)
        return {
            "iterations": [
                {
                    "step": 1,
                    "llm_output": raw_output,
                    "execution_output": execution["stdout"],
                    "error": execution["stderr"],
                    "success": execution["success"],
                }
            ],
            "final_success": execution["success"],
            "final_code": candidate_code,
        }


def evaluate_cases(cases: List[DebugCase], llm: UnifiedLLM, agent_iterations: int = 4) -> Dict[str, Any]:
    baseline_runner = BaselineRunner(llm)
    agent_runner = DebugAgent(llm=llm, max_iterations=agent_iterations, verbose=True)

    full_traces: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for case in cases:
        baseline_trace = baseline_runner.run_case(case)
        baseline_record = {
            "case_id": case.id,
            "setup": "baseline",
            **baseline_trace,
        }
        baseline_record["failure_type"] = classify_failure(baseline_record)
        full_traces.append(baseline_record)
        rows.append(
            {
                "case_id": case.id,
                "setup": "baseline",
                "final_success": baseline_record["final_success"],
                "iterations_used": len(baseline_record["iterations"]),
                "failure_type": baseline_record["failure_type"],
            }
        )

        agent_trace = agent_runner.run_case(
            {
                "id": case.id,
                "buggy_code": case.buggy_code,
                "task": case.task,
                "test_code": case.test_code,
            }
        )
        agent_record = {
            "case_id": case.id,
            "setup": "agent",
            **agent_trace,
        }
        agent_record["failure_type"] = classify_failure(agent_record)
        full_traces.append(agent_record)
        rows.append(
            {
                "case_id": case.id,
                "setup": "agent",
                "final_success": agent_record["final_success"],
                "iterations_used": len(agent_record["iterations"]),
                "failure_type": agent_record["failure_type"],
            }
        )

    metrics = compute_metrics(full_traces)
    return {
        "full_traces": full_traces,
        "results_rows": rows,
        "metrics": metrics,
    }


def compute_metrics(full_traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {"baseline": [], "agent": []}
    for trace in full_traces:
        grouped[trace["setup"]].append(trace)

    metrics: Dict[str, Any] = {}
    for setup, traces in grouped.items():
        success_rate = mean([1.0 if trace["final_success"] else 0.0 for trace in traces])
        failures = Counter(trace["failure_type"] for trace in traces if trace["failure_type"] != "none")
        metrics[setup] = {
            "success_rate": success_rate,
            "failure_types": {
                "error_misinterpretation": failures.get("error_misinterpretation", 0),
                "wrong_fix": failures.get("wrong_fix", 0),
                "runtime_error": failures.get("runtime_error", 0),
                "loop_failure": failures.get("loop_failure", 0),
            },
        }

    agent_iterations = [
        len(trace["iterations"])
        for trace in grouped["agent"]
    ]
    metrics["agent"]["average_iterations"] = mean(agent_iterations)
    metrics["improvement"] = metrics["agent"]["success_rate"] - metrics["baseline"]["success_rate"]
    return metrics
