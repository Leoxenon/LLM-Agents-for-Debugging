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
    improvement = results["metrics"]["improvement"]

    print(f"Baseline success rate: {baseline_success:.3f}")
    print(f"Agent success rate: {agent_success:.3f}")
    print(f"Improvement: {improvement:.3f}")


if __name__ == "__main__":
    main()
