"""Fault parameter definitions for scenario-based testing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import math


@dataclass
class PumpFaultParams:
    vol_eff_multiplier: float = 1.0
    leakage_multiplier: float = 1.0
    mech_eff_multiplier: float = 1.0


@dataclass
class CylinderFaultParams:
    internal_leak_coeff: float = 0.0
    external_leak_coeff: float = 0.0
    friction_multiplier: float = 1.0
    drift_velocity_m_s: float = 0.0


@dataclass
class ValveFaultParams:
    deadband_offset: float = 0.0
    gain_reduction: float = 0.0


@dataclass
class SensorFaultParams:
    pressure_bias_pa: float = 0.0
    pressure_scale: float = 1.0
    position_bias_m: float = 0.0
    position_scale: float = 1.0


@dataclass
class FaultConfig:
    pump: Optional[PumpFaultParams] = None
    cylinders: Dict[str, CylinderFaultParams] = field(default_factory=dict)
    valves: Dict[str, ValveFaultParams] = field(default_factory=dict)
    sensors: Dict[str, SensorFaultParams] = field(default_factory=dict)

    def to_multilabel(self) -> Dict[str, bool]:
        labels: Dict[str, bool] = {
            "pump": False,
            "boom": False,
            "arm": False,
            "bucket": False,
        }
        if self.pump is not None:
            labels["pump"] = True
        for name in self.cylinders:
            if name in labels:
                labels[name] = True
        return labels
