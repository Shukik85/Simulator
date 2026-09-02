from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class LSPumpConfig:
    max_displacement_cc_rev: float = 70.0
    max_speed_rpm: float = 2200.0
    margin_standby_bar: float = 18.0
    margin_max_bar: float = 30.0
    response_time_s: float = 0.1
    vol_eff_0: float = 0.95
    vol_eff_kp: float = 0.10
    mech_eff_0: float = 0.92
    mech_eff_kp: float = 0.08
    case_leak_k: float = 2.0e-12
    power_hp: float = 120.0


@dataclass(frozen=True)
class LSValveSectionConfig:
    q_nom_lpm_at_dp: float = 60.0
    delta_p_rated_bar: float = 20.0
    deadband: float = 0.02
    flow_exp: float = 1.0
    cd: float = 0.65
    drain_gain: float = 5.0
    spool_tau_s: float = 0.05
    clamp_hyst: float = 0.05
    anti_cav_v_makeup: float = 0.05


@dataclass(frozen=True)
class ReliefValveConfig:
    crack_bar: float = 300.0
    gain_m3_s_per_bar: float = 5.0e-5
    max_bar: float = 350.0


@dataclass(frozen=True)
class CylinderDynamicsConfig:
    dead_volume_m3: float = 1.0e-4
    line_volume_m3: float = 5.0e-4
    visc_damping: float = 120.0
    coulomb_friction: float = 600.0
    mass_equiv: float = 800.0


@dataclass(frozen=True)
class FluidConfig:
    rho: float = 850.0
    bulk_modulus: float = 1.7e9
    cp: float = 1900.0


@dataclass(frozen=True)
class LSConfig:
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
