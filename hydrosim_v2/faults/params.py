from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

_EPS = 1e-12


@dataclass
class CylinderFaultParams:
    """Fault parameters for one hydraulic cylinder."""
    internal_leak_coeff: float = 0.0   # m³/(s·Pa) — piston seal leakage
    external_leak_coeff: float = 0.0   # m³/(s·Pa) — rod seal to tank
    friction_multiplier: float = 1.0   # × nominal friction
    area_factor: float = 1.0           # × nominal area (blockage)


@dataclass
class PumpFaultParams:
    """Fault parameters for the LS pump."""
    vol_eff_multiplier: float = 1.0    # × nominal volumetric efficiency
    leakage_multiplier: float = 1.0    # × nominal case leakage
    margin_multiplier: float = 1.0     # × nominal LS margin


@dataclass
class ValveFaultParams:
    """Fault parameters for one valve section."""
    flow_multiplier: float = 1.0       # × nominal flow gain
    bias: float = 0.0                  # spool position offset


@dataclass
class SensorFaultParams:
    """Fault parameters for sensor readings (applied during recording)."""
    pressure_bias_pa: float = 0.0
    pressure_scale: float = 1.0
    position_bias_m: float = 0.0
    position_scale: float = 1.0
    flow_bias_m3s: float = 0.0
    flow_scale: float = 1.0


@dataclass
class FaultConfig:
    """Complete fault configuration for a simulation run."""
    pump: PumpFaultParams = field(default_factory=PumpFaultParams)
    valves: Dict[str, ValveFaultParams] = field(default_factory=lambda: {
        n: ValveFaultParams() for n in ["boom", "arm", "bucket"]
    })
    cylinders: Dict[str, CylinderFaultParams] = field(default_factory=lambda: {
        n: CylinderFaultParams() for n in ["boom", "arm", "bucket"]
    })
    sensors: Dict[str, SensorFaultParams] = field(default_factory=lambda: {
        n: SensorFaultParams() for n in [
            "p_pump", "p_boom_A", "p_boom_B", "p_arm_A", "p_arm_B",
            "p_bucket_A", "p_bucket_B",
            "x_boom", "x_arm", "x_bucket",
            "Q_pump", "Q_boom", "Q_arm", "Q_bucket",
        ]
    })

    def to_multilabel(self, threshold: float = 0.01) -> Dict[str, bool]:
        """Convert continuous fault params to binary multi-label vector.

        Labels: pump, boom, arm, bucket.
        A label is True if any fault parameter for that component exceeds threshold.
        """
        labels = {
            "pump": any(abs(v) > threshold for v in [
                self.pump.vol_eff_multiplier - 1.0,
                self.pump.leakage_multiplier - 1.0,
                self.pump.margin_multiplier - 1.0,
            ]),
            "boom": _cyl_fault_active(self.cylinders.get("boom"), threshold),
            "arm": _cyl_fault_active(self.cylinders.get("arm"), threshold),
            "bucket": _cyl_fault_active(self.cylinders.get("bucket"), threshold),
        }
        return labels


def _cyl_fault_active(cfg: CylinderFaultParams | None, th: float) -> bool:
    if cfg is None:
        return False
    return any(abs(v) > th for v in [
        cfg.internal_leak_coeff,
        cfg.external_leak_coeff,
        cfg.friction_multiplier - 1.0,
        cfg.area_factor - 1.0,
    ])
