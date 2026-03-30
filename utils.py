import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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
    api_key = os.getenv("API_KEY", "").strip()
    if not api_key:
        api_key = os.getenv("ARK_API_KEY", "").strip()

    model_name = os.getenv("MODEL_NAME", "").strip()
    if not model_name:
        model_name = os.getenv("ARK_ENDPOINT_ID", "").strip()

    return {
        "api_key": api_key,
        "base_url": os.getenv("BASE_URL", "").strip(),
        "model_name": model_name,
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
    final_evaluation = trace.get("final_evaluation", {})
    final_error = normalize_error(final_evaluation.get("stderr", ""))
    errors = [normalize_error(item.get("error", "")) for item in iterations if item.get("error")]
    if final_error:
        errors.append(final_error)

    llm_outputs = [
        item.get("candidate_code", "") or extract_python_code(item.get("llm_output", ""))
        for item in iterations
    ]

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


def read_json(path: str, default: Any = None) -> Any:
    """Read JSON from disk; return default if the file does not exist."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def write_json_once(path: str, payload: Any) -> None:
    """Write JSON to `path` only if it does not already exist.

    This is a lightweight guard to prevent accidental overwrites of 'final' artifacts.
    """
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing JSON artifact: {path}")
    write_json(path, payload)


def validate_candidate_code(candidate_code: str, required_substrings: Optional[List[str]] = None) -> List[str]:
    """Return list of validation error messages (empty means OK)."""
    errors: List[str] = []
    if not candidate_code or not candidate_code.strip():
        errors.append("empty_candidate_code")
        return errors

    required_substrings = required_substrings or []
    for token in required_substrings:
        if token not in candidate_code:
            errors.append(f"missing:{token}")

    return errors
