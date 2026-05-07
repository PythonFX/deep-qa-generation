from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class JSONLLM(ABC):
    @abstractmethod
    def run_json(self, prompt_name: str, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

