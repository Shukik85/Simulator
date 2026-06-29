from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from hydrosim_v2.core._validators import check_positive, check_range


@dataclass(frozen=True)
class LSPumpConfig:
    """Axial piston pump with LS control."""
    max_displacement_cc_rev: float = 70.0
    max_speed_rpm: float = 2200.0
    margin_standby_bar: float = 18.0   # ΔP = Ppump - PLS on standby
    margin_max_bar: float = 30.0       # margin at full stroke
    response_time_s: float = 0.1       # swash plate time constant
    vol_eff_0: float = 0.95
    vol_eff_kp: float = 0.10
    mech_eff_0: float = 0.92
    mech_eff_kp: float = 0.08
    case_leak_k: float = 2.0e-12

    def __post_init__(self) -> None:
        check_positive("max_displacement_cc_rev", self.max_displacement_cc_rev)
        check_positive("max_speed_rpm", self.max_speed_rpm)
        check_positive("margin_standby_bar", self.margin_standby_bar)
        check_positive("margin_max_bar", self.margin_max_bar)


@dataclass(frozen=True)
class LSValveSectionConfig:
    """One LS valve section (spool + compensator)."""
    q_nom_lpm_at_dp: float = 60.0   # nominal flow at rated ΔP
    delta_p_rated_bar: float = 20.0  # rated compensator ΔP (14-20 bar typical)
    deadband: float = 0.02
    flow_exp: float = 1.0
    cd: float = 0.65

    def __post_init__(self) -> None:
        check_positive("q_nom_lpm_at_dp", self.q_nom_lpm_at_dp)
        check_positive("delta_p_rated_bar", self.delta_p_rated_bar)
        check_range("deadband", self.deadband, 0.0, 0.5)
        check_range("flow_exp", self.flow_exp, 0.5, 3.0)


@dataclass(frozen=True)
class ReliefValveConfig:
    crack_bar: float = 300.0
    gain_m3_s_per_bar: float = 1.0e-5
    max_bar: float = 350.0


@dataclass(frozen=True)
class CylinderDynamicsConfig:
    """Hydraulic + friction params for a cylinder."""
    dead_volume_m3: float = 1.0e-4
    line_volume_m3: float = 5.0e-4
    visc_damping: float = 120.0      # N·s/m
    coulomb_friction: float = 600.0  # N
    mass_equiv: float = 800.0        # kg (equivalent reflected mass)


@dataclass(frozen=True)
class FluidConfig:
    rho: float = 850.0           # kg/m³
    bulk_modulus: float = 1.7e9  # Pa
    cp: float = 1900.0           # J/(kg·K)


@dataclass(frozen=True)
class LSConfig:
    """Full LS hydraulic system configuration (no swing, no travel)."""
    pump: LSPumpConfig = field(default_factory=LSPumpConfig)
    relief: ReliefValveConfig = field(default_factory=ReliefValveConfig)
    valve_sections: Dict[str, LSValveSectionConfig] = field(default_factory=lambda: {
        "boom": LSValveSectionConfig(q_nom_lpm_at_dp=60.0),
        "arm": LSValveSectionConfig(q_nom_lpm_at_dp=60.0),
        "bucket": LSValveSectionConfig(q_nom_lpm_at_dp=40.0),
    })
    cylinder_dynamics: Dict[str, CylinderDynamicsConfig] = field(default_factory=lambda: {
        "boom": CylinderDynamicsConfig(mass_equiv=800.0),
        "arm": CylinderDynamicsConfig(mass_equiv=600.0),
        "bucket": CylinderDynamicsConfig(mass_equiv=400.0),
    })
    fluid: FluidConfig = field(default_factory=FluidConfig)
