"""Конфиги гидросимулятора.

В репозитории исторически есть модуль `hydrosim/config.py`.
Одновременно нужен пакет `hydrosim.config.*` для разнесённых конфигов.

Чтобы не ломать старый код:
- новые механические конфиги доступны в `hydrosim.config.mechanics`;
- если существует `hydrosim/config.py`, он загружается как legacy-модуль;
- legacy-символы реэкспортируются, но при конфликте имён создаётся алиас
  `Legacy<Имя>` (например, `LegacyMechanicsConfig`).
"""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

from .mechanics import (  # noqa: F401
    DEFAULT_MECHANICS_CONFIG,
    CylinderGeometry,
    LinkGeometry,
    MechanicsConfig,
)


def _load_legacy_config() -> Any | None:
    """Load legacy module `hydrosim/config.py`.

    NOTE:
    - We must register the module in sys.modules *before* exec_module.
      Python 3.14 dataclasses may access sys.modules[cls.__module__] during
      class processing; without registration this can crash test collection.
    """

    legacy_path = Path(__file__).resolve().parents[1] / "config.py"
    if not legacy_path.exists():
        return None

    spec = spec_from_file_location("hydrosim._legacy_config", legacy_path)
    if spec is None or spec.loader is None:
        return None

    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_legacy = _load_legacy_config()

_legacy_exported: list[str] = []

if _legacy is not None:
    for _name in (
        "FluidConfig",
        "PumpConfig",
        "OpenCenterConfig",
        "ReliefValveConfig",
        "ValveSectionConfig",
        "ValveBankConfig",
        "CylinderConfig",
        "SwingMotorConfig",
        "MechanicsConfig",
        "SoilConfig",
        "ThermalConfig",
        "SimulationConfig",
        "SensorConfig",
        "SystemConfig",
    ):
        if not hasattr(_legacy, _name):
            continue

        _obj = getattr(_legacy, _name)

        if _name in globals():
            alias = f"Legacy{_name}"
            globals()[alias] = _obj
            _legacy_exported.append(alias)
        else:
            globals()[_name] = _obj
            _legacy_exported.append(_name)


__all__ = [
    "LinkGeometry",
    "CylinderGeometry",
    "MechanicsConfig",
    "DEFAULT_MECHANICS_CONFIG",
    *_legacy_exported,
]
