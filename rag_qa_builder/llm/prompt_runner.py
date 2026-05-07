from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_qa_builder.config import AppConfig
from rag_qa_builder.llm.openai_client import CachedLLMClient


class PromptRunner:
    def __init__(self, config: AppConfig, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self.client = CachedLLMClient(config.llm, output_dir / ".cache" / "llm_calls") if config.llm.enabled else None

    def run_json(self, prompt_name: str, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.client:
            raise RuntimeError("LLM client is disabled")
        return self.client.run_json(prompt_name=prompt_name, system=system, payload=payload)

    def maybe_run_json(self, prompt_name: str, system: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self.client:
            return None
        try:
            return self.run_json(prompt_name=prompt_name, system=system, payload=payload)
        except Exception:
            return None

