from __future__ import annotations

from dataclasses import dataclass, field

from hydrosim_v2.config.base import (
    Attachment2D, LinkGeometry, CylinderGeometry,
    BucketLeverMechanismParams, BodyGeometry,
)
from hydrosim_v2.config.mechanics import (
    ExcavatorMechanicsConfig, DEFAULT_MECHANICS_CONFIG,
)
from hydrosim_v2.config.hydraulics import (
    LSPumpConfig, LSValveSectionConfig, ReliefValveConfig,
    CylinderDynamicsConfig, FluidConfig, LSConfig,
)


@dataclass(frozen=True)
class SoilConfig:
    base_resistance_N: float = 25000.0
    vel_gain_N_per_m_s: float = 8000.0
    penetration_gain_N: float = 60000.0
    randomness: float = 0.25


@dataclass(frozen=True)
class ExcavatorConfig:
    mechanics: ExcavatorMechanicsConfig
    hydraulics: LSConfig
    soil: SoilConfig = field(default_factory=SoilConfig)


__all__ = [
    # Base types
    "Attachment2D", "LinkGeometry", "CylinderGeometry",
    "BucketLeverMechanismParams", "BodyGeometry",
    "ExcavatorMechanicsConfig", "DEFAULT_MECHANICS_CONFIG",
    # Hydraulics
    "LSPumpConfig", "LSValveSectionConfig", "ReliefValveConfig",
    "CylinderDynamicsConfig", "FluidConfig", "LSConfig",
    # Top-level
    "SoilConfig", "ExcavatorConfig",
]
