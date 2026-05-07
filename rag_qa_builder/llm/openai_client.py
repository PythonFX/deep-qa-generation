from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tenacity import retry, stop_after_attempt, wait_fixed

from rag_qa_builder.config import LLMConfig
from rag_qa_builder.llm.base import JSONLLM
from rag_qa_builder.utils.json_utils import dump_json, load_json

from llm_client.llm_factory import create_anthropic_client


class CachedLLMClient(JSONLLM):
    def __init__(self, config: LLMConfig, cache_dir: Path) -> None:
        self.config = config
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = self._build_client() if config.enabled else None

    def _build_client(self) -> Any:
        return create_anthropic_client(
            profile_name=self.config.profile,
            timeout=float(self.config.request_timeout_seconds)
        )

    def run_json(self, prompt_name: str, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._client:
            raise RuntimeError("LLM disabled")
        key = hashlib.sha1(
            json.dumps(
                {
                    "prompt_name": prompt_name,
                    "model": self.config.model,
                    "profile": self.config.profile,
                    "payload": payload,
                    "version": self.config.prompt_version,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            return load_json(cache_path)
        response = self._request_with_retry(system=system, payload=payload)
        parsed = _extract_json(response.content)
        dump_json(cache_path, parsed)
        return parsed

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def _request_with_retry(self, system: str, payload: dict[str, Any]) -> Any:
        from llm_client.models import Message
        return self._client.completion(
            provider=self.config.profile,
            model=self.config.model,
            system=system,
            messages=[Message(role="user", content=json.dumps(payload, ensure_ascii=False))],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(stripped[start:end + 1])
    raise ValueError("LLM response is not valid JSON")
