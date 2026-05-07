from __future__ import annotations

import time
from typing import Any, Callable


def stop_after_attempt(attempts: int) -> int:
    return attempts


def wait_fixed(seconds: float) -> float:
    return seconds


def retry(stop: int, wait: float) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error: Exception | None = None
            for attempt in range(stop):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # pragma: no cover - passthrough behavior
                    last_error = exc
                    if attempt < stop - 1 and wait:
                        time.sleep(wait)
            if last_error is not None:
                raise last_error
            return func(*args, **kwargs)
        return wrapper
    return decorator

