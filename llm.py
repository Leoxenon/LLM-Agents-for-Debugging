from dataclasses import dataclass
from typing import Optional

from utils import extract_python_code, get_env_config

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - handled at runtime
    ChatOpenAI = None
    HumanMessage = None
    SystemMessage = None


SYSTEM_PROMPT = """You are an expert Python debugging assistant.
You will receive buggy Python code and a task description.
Return only the complete corrected Python code.
Do not include explanations outside the code block.
Preserve the function name and provide executable code.
"""


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model_name: str
    temperature: float = 0.0


class UnifiedLLM:
    def __init__(self, config: Optional[LLMConfig] = None):
        env_config = get_env_config()
        self.config = config or LLMConfig(
            api_key=env_config["api_key"],
            base_url=env_config["base_url"],
            model_name=env_config["model_name"],
        )

    def is_configured(self) -> bool:
        return all([self.config.api_key, self.config.base_url, self.config.model_name])

    def _build_model(self):
        if ChatOpenAI is None:
            raise RuntimeError(
                "LangChain dependencies are missing. Install requirements.txt before running the experiment."
            )
        if not self.is_configured():
            raise RuntimeError(
                "Missing LLM configuration. Please set API_KEY, BASE_URL, and MODEL_NAME."
            )
        return ChatOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            model=self.config.model_name,
            temperature=self.config.temperature,
        )

    def invoke(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        model = self._build_model()
        response = model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt),
            ]
        )
        return response.content if isinstance(response.content, str) else str(response.content)

    def build_fix_prompt(self, buggy_code: str, task: str) -> str:
        return (
            f"Task:\n{task}\n\n"
            f"Buggy code:\n```python\n{buggy_code}\n```\n\n"
            "Return the full corrected Python code only."
        )

    def generate_fix(self, buggy_code: str, task: str) -> str:
        prompt = self.build_fix_prompt(buggy_code, task)
        return extract_python_code(self.invoke(prompt))
