from dataset import load_dataset
from evaluator import evaluate_cases
from llm import UnifiedLLM
from utils import write_csv, write_json


def main() -> None:
    llm = UnifiedLLM()
    cases = load_dataset()
    results = evaluate_cases(cases, llm)

    write_json("full_traces.json", results["full_traces"])
    write_csv(
        "results.csv",
        results["results_rows"],
        fieldnames=["case_id", "setup", "final_success", "iterations_used", "failure_type"],
    )
    write_json("metrics.json", results["metrics"])

    baseline_success = results["metrics"]["baseline"]["success_rate"]
    agent_success = results["metrics"]["agent"]["success_rate"]
    reflection_success = results["metrics"]["agent_reflection"]["success_rate"]
    comparisons = results["metrics"]["comparisons"]

    print(f"Baseline success rate: {baseline_success:.3f}")
    print(f"Agent success rate: {agent_success:.3f}")
    print(f"Self-reflection agent success rate: {reflection_success:.3f}")
    print(f"Agent vs baseline improvement: {comparisons['agent_vs_baseline_improvement']:.3f}")
    print(
        "Self-reflection agent vs baseline improvement: "
        f"{comparisons['agent_reflection_vs_baseline_improvement']:.3f}"
    )
    print(
        "Self-reflection agent vs agent improvement: "
        f"{comparisons['agent_reflection_vs_agent_improvement']:.3f}"
    )


if __name__ == "__main__":
    main()
