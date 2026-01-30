"""Конфиги гидросимулятора.

Пакет `hydrosim.config` содержит:
- системные конфиги (масла/насоса/клапанов/цилиндров/сенсоров) в `hydrosim.config.models`;
- механическую геометрию в `hydrosim.config.mechanics`.

Рекомендация по импортам:
- from hydrosim.config.models import SystemConfig
- from hydrosim.config.mechanics import DEFAULT_MECHANICS_CONFIG
"""

from __future__ import annotations

from .mechanics import (  # noqa: F401
    DEFAULT_MECHANICS_CONFIG,
    CylinderGeometry,
    LinkGeometry,
    MechanicsConfig,
)
from .models import (  # noqa: F401
    CylinderConfig,
    FluidConfig,
    MassPropertiesConfig,
    OpenCenterConfig,
    PumpConfig,
    ReliefValveConfig,
    SensorConfig,
    SimulationConfig,
    SoilConfig,
    SwingMotorConfig,
    SystemConfig,
    ThermalConfig,
    ValveBankConfig,
    ValveSectionConfig,
)

__all__ = [
    # Mechanics geometry
    "LinkGeometry",
    "CylinderGeometry",
    "MechanicsConfig",
    "DEFAULT_MECHANICS_CONFIG",
    # System configs
    "FluidConfig",
    "PumpConfig",
    "OpenCenterConfig",
    "ReliefValveConfig",
    "ValveSectionConfig",
    "ValveBankConfig",
    "CylinderConfig",
    "SwingMotorConfig",
    "MassPropertiesConfig",
    "SoilConfig",
    "ThermalConfig",
    "SimulationConfig",
    "SensorConfig",
    "SystemConfig",
]
