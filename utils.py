import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


def ensure_text_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str, payload: Any) -> None:
    ensure_text_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def write_csv(path: str, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    ensure_text_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def get_env_config() -> Dict[str, str]:
    return {
        "api_key": os.getenv("API_KEY", ""),
        "base_url": os.getenv("BASE_URL", ""),
        "model_name": os.getenv("MODEL_NAME", ""),
    }


def extract_python_code(text: str) -> str:
    stripped = text.strip()
    fenced = re.findall(r"```python\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[-1].strip() + "\n"

    generic = re.findall(r"```\s*(.*?)```", stripped, flags=re.DOTALL)
    if generic:
        return generic[-1].strip() + "\n"

    return stripped + ("\n" if stripped and not stripped.endswith("\n") else "")


def normalize_error(error: str) -> str:
    return (error or "").strip()


def classify_failure(trace: Dict[str, Any]) -> str:
    if trace.get("final_success"):
        return "none"

    iterations = trace.get("iterations", [])
    errors = [normalize_error(item.get("error", "")) for item in iterations if item.get("error")]
    llm_outputs = [extract_python_code(item.get("llm_output", "")) for item in iterations]

    if len(llm_outputs) >= 2 and len(set(output.strip() for output in llm_outputs if output.strip())) == 1:
        return "loop_failure"

    joined_errors = "\n".join(errors).lower()
    if any(token in joined_errors for token in ["traceback", "typeerror", "nameerror", "zerodivisionerror", "unboundlocalerror", "recursionerror", "syntaxerror", "assertionerror"]):
        if len(iterations) > 1 and errors:
            return "error_misinterpretation"
        return "runtime_error"

    return "wrong_fix"


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
