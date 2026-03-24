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
        return {
            "iterations": [
                {
                    "step": 1,
                    "llm_output": raw_output,
                    "candidate_code": candidate_code,
                    "execution_output": "",
                    "error": "",
                    "success": None,
                }
            ],
            "final_code": candidate_code,
        }


def evaluate_cases(cases: List[DebugCase], llm: UnifiedLLM, agent_iterations: int = 4) -> Dict[str, Any]:
    baseline_runner = BaselineRunner(llm)
    agent_runner = DebugAgent(
        llm=llm,
        max_iterations=agent_iterations,
        verbose=True,
        use_reflection=False,
        setup_name="agent",
    )
    reflection_agent_runner = DebugAgent(
        llm=llm,
        max_iterations=agent_iterations,
        verbose=True,
        use_reflection=True,
        setup_name="agent_reflection",
    )

    full_traces: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for case in cases:
        baseline_trace = baseline_runner.run_case(case)
        baseline_record = {
            "case_id": case.id,
            "setup": "baseline",
            **baseline_trace,
        }
        baseline_evaluation = run_python_code(baseline_record["final_code"] + "\n" + case.test_code)
        baseline_record["final_evaluation"] = baseline_evaluation
        baseline_record["final_success"] = baseline_evaluation["success"]
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
        agent_evaluation = run_python_code(agent_record["final_code"] + "\n" + case.test_code)
        agent_record["final_evaluation"] = agent_evaluation
        agent_record["final_success"] = agent_evaluation["success"]
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

        reflection_agent_trace = reflection_agent_runner.run_case(
            {
                "id": case.id,
                "buggy_code": case.buggy_code,
                "task": case.task,
                "test_code": case.test_code,
            }
        )
        reflection_agent_record = {
            "case_id": case.id,
            "setup": "agent_reflection",
            **reflection_agent_trace,
        }
        reflection_agent_evaluation = run_python_code(
            reflection_agent_record["final_code"] + "\n" + case.test_code
        )
        reflection_agent_record["final_evaluation"] = reflection_agent_evaluation
        reflection_agent_record["final_success"] = reflection_agent_evaluation["success"]
        reflection_agent_record["failure_type"] = classify_failure(reflection_agent_record)
        full_traces.append(reflection_agent_record)
        rows.append(
            {
                "case_id": case.id,
                "setup": "agent_reflection",
                "final_success": reflection_agent_record["final_success"],
                "iterations_used": len(reflection_agent_record["iterations"]),
                "failure_type": reflection_agent_record["failure_type"],
            }
        )

    metrics = compute_metrics(full_traces)
    return {
        "full_traces": full_traces,
        "results_rows": rows,
        "metrics": metrics,
    }


def compute_metrics(full_traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for trace in full_traces:
        grouped.setdefault(trace["setup"], []).append(trace)

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

    for setup in ["agent", "agent_reflection"]:
        if setup in grouped:
            metrics[setup]["average_iterations"] = mean(
                [len(trace["iterations"]) for trace in grouped[setup]]
            )

    baseline_success = metrics.get("baseline", {}).get("success_rate", 0.0)
    agent_success = metrics.get("agent", {}).get("success_rate", 0.0)
    reflection_success = metrics.get("agent_reflection", {}).get("success_rate", 0.0)
    metrics["comparisons"] = {
        "agent_vs_baseline_improvement": agent_success - baseline_success,
        "agent_reflection_vs_baseline_improvement": reflection_success - baseline_success,
        "agent_reflection_vs_agent_improvement": reflection_success - agent_success,
    }
    return metrics
