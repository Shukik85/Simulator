"""Конфиги гидросимулятора.

ВНИМАНИЕ: в репозитории уже существует файл `hydrosim/config.py`.
Создание пакета `hydrosim.config` может изменить семантику импорта `hydrosim.config`.

Чтобы не ломать существующий код, этот пакет:
- предоставляет новые механические конфиги в `hydrosim.config.mechanics`;
- при наличии `hydrosim/config.py` загружает его как legacy-модуль и реэкспортирует
  основные символы (если они существуют).
"""

from __future__ import annotations

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
    legacy_path = Path(__file__).resolve().parents[1] / "config.py"
    if not legacy_path.exists():
        return None

    spec = spec_from_file_location("hydrosim._legacy_config", legacy_path)
    if spec is None or spec.loader is None:
        return None

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_legacy = _load_legacy_config()

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
        if hasattr(_legacy, _name):
            globals()[_name] = getattr(_legacy, _name)


__all__ = [
    "LinkGeometry",
    "CylinderGeometry",
    "MechanicsConfig",
    "DEFAULT_MECHANICS_CONFIG",
]
