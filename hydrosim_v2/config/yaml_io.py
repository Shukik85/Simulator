from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml

from hydrosim_v2.config import ExcavatorConfig

T = TypeVar("T")


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert frozen dataclass to plain dict for YAML."""
    if hasattr(obj, "__dataclass_fields__"):
        result = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            result[field_name] = _dataclass_to_dict(value)
        return result
    if isinstance(obj, tuple) and hasattr(obj, "__len__"):
        return list(obj)
    return obj


def _dict_to_dataclass(data: dict, cls: type) -> Any:
    """Recursively build frozen dataclass from dict."""
    from dataclasses import fields

    field_defs = {f.name: f.type for f in fields(cls)}
    kwargs = {}
    for name, value in data.items():
        if name not in field_defs:
            continue
        ftype = field_defs[name]
        if hasattr(ftype, "__dataclass_fields__"):
            kwargs[name] = _dict_to_dataclass(value, ftype)
        elif isinstance(ftype, type) and ftype.__name__ == "dict" and isinstance(value, dict):
            kwargs[name] = value
        else:
            kwargs[name] = value
    return cls(**kwargs)


def save_yaml(cfg: ExcavatorConfig, path: str | Path) -> None:
    data = _dataclass_to_dict(cfg)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_yaml(path: str | Path) -> ExcavatorConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return _dict_to_dataclass(data, ExcavatorConfig)
