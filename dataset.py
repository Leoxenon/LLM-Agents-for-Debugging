import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class DebugCase:
    id: str
    buggy_code: str
    task: str
    test_code: str


def load_dataset(path: str = "dataset.json") -> List[DebugCase]:
    dataset_path = Path(path)
    raw_cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    return [DebugCase(**item) for item in raw_cases]
