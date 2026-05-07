from __future__ import annotations

from copy import deepcopy
from dataclasses import MISSING
from typing import Any, get_args, get_origin, get_type_hints


class FieldInfo:
    def __init__(self, default: Any = MISSING, default_factory: Any = MISSING, **_: Any) -> None:
        self.default = default
        self.default_factory = default_factory

    def get_default(self) -> Any:
        if self.default_factory is not MISSING:
            return self.default_factory()
        if self.default is not MISSING:
            return deepcopy(self.default)
        return None


def Field(default: Any = MISSING, default_factory: Any = MISSING, **kwargs: Any) -> Any:
    return FieldInfo(default=default, default_factory=default_factory, **kwargs)


class BaseModel:
    def __init__(self, **kwargs: Any) -> None:
        annotations = _all_annotations(self.__class__)
        for name, annotation in annotations.items():
            if name in kwargs:
                value = _coerce(annotation, kwargs[name])
            else:
                value = _default_for(self.__class__, name)
            setattr(self, name, value)

    @classmethod
    def model_validate(cls, data: Any) -> Any:
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict for {cls.__name__}, got {type(data)!r}")
        return cls(**data)

    def model_dump(self, mode: str | None = None) -> dict[str, Any]:
        return {name: _dump(getattr(self, name)) for name in _all_annotations(self.__class__)}


def _all_annotations(cls: type) -> dict[str, Any]:
    annotations: dict[str, Any] = {}
    for base in reversed(cls.__mro__):
        try:
            annotations.update(get_type_hints(base))
        except Exception:
            annotations.update(getattr(base, "__annotations__", {}))
    return annotations


def _default_for(cls: type, name: str) -> Any:
    value = getattr(cls, name, MISSING)
    if isinstance(value, FieldInfo):
        return value.get_default()
    if value is MISSING:
        return None
    return deepcopy(value)


def _coerce(annotation: Any, value: Any) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if value is None:
        return None
    if origin is list and args:
        return [_coerce(args[0], item) for item in value]
    if origin is dict and len(args) == 2:
        return {key: _coerce(args[1], item) for key, item in value.items()}
    if origin is tuple and args:
        return tuple(_coerce(args[0], item) for item in value)
    if origin is not None and args:
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1:
            return _coerce(non_none[0], value)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel) and isinstance(value, dict):
        return annotation.model_validate(value)
    return value


def _dump(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, list):
        return [_dump(item) for item in value]
    if isinstance(value, dict):
        return {key: _dump(item) for key, item in value.items()}
    return value
