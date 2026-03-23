import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict


def run_python_code(code: str, timeout_seconds: int = 5) -> Dict[str, object]:
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "candidate.py"
        script_path.write_text(code, encoding="utf-8")

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=tmpdir,
        )

        success = result.returncode == 0
        return {
            "success": success,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }


def render_execution_feedback(execution_result: Dict[str, object]) -> str:
    return json.dumps(
        {
            "success": execution_result["success"],
            "stdout": execution_result["stdout"],
            "stderr": execution_result["stderr"],
            "returncode": execution_result["returncode"],
        },
        indent=2,
        ensure_ascii=True,
    )
