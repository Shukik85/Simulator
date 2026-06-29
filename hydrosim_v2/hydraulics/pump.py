from __future__ import annotations

import math

from hydrosim_v2.config.hydraulics import LSPumpConfig
from hydrosim_v2.core.units import bar_to_Pa, Pa_to_bar

_EPS = 1e-12


class LSPumpModel:
    """Load-sensing axial piston pump with swashplate dynamics."""

    def __init__(self, config: LSPumpConfig, speed_rpm: float | None = None):
        self.cfg = config
        self.n_rpm = speed_rpm if speed_rpm is not None else config.max_speed_rpm

    @property
    def max_theoretical_flow_m3s(self) -> float:
        """Maximum theoretical flow at max displacement and rated speed (m³/s)."""
        Vg_m3 = self.cfg.max_displacement_cc_rev * 1e-6  # cc/rev → m³/rev
        return Vg_m3 * self.n_rpm / 60.0

    def target_displacement(
        self,
        p_pump: float,
        p_ls_max: float,
    ) -> float:
        """Target displacement fraction ε ∈ [0, 1] from LS controller.

        The controller seeks p_pump - p_ls_max = Δp_LS.
        If pump pressure > cutoff, destroke.
        """
        p_stby = bar_to_Pa(self.cfg.margin_standby_bar)
        cutoff = bar_to_Pa(self.cfg.margin_max_bar)

        if p_pump > cutoff:
            return 0.0

        delta = p_pump - p_ls_max

        if delta < p_stby:
            return 0.0
        elif delta > cutoff:
            return 1.0

        eps = (delta - p_stby) / (cutoff - p_stby)
        return float(max(0.0, min(1.0, eps)))

    def flow_m3s(self, epsilon: float, vol_eff: float | None = None) -> float:
        """Actual pump flow at given displacement fraction ε."""
        if vol_eff is None:
            vol_eff = self.volumetric_efficiency(epsilon * self.max_theoretical_flow_m3s)
        return epsilon * self.max_theoretical_flow_m3s * vol_eff

    def volumetric_efficiency(self, Q_th: float) -> float:
        """Volumetric efficiency as function of theoretical flow."""
        eff = self.cfg.vol_eff_0 - self.cfg.vol_eff_kp * Q_th / (self.max_theoretical_flow_m3s + _EPS)
        return float(max(0.5, min(1.0, eff)))

    def mechanical_efficiency(self, pressure_bar: float, epsilon: float) -> float:
        eff = self.cfg.mech_eff_0 - self.cfg.mech_eff_kp * (1.0 - epsilon)
        return float(max(0.5, min(1.0, eff)))

    def case_leak_flow_m3s(self, p_pump: float) -> float:
        """Return case leakage flow (m³/s)."""
        return self.cfg.case_leak_k * p_pump

    def swash_plate_rhs(
        self,
        epsilon: float,
        p_pump: float,
        p_ls_max: float,
    ) -> float:
        """Time derivative of swash-plate position dε/dt."""
        target = self.target_displacement(p_pump, p_ls_max)
        return (target - epsilon) / (self.cfg.response_time_s + _EPS)

    def flow_to_system_rhs(
        self,
        p_pump: float,
        epsilon: float,
        total_valve_flow_m3s: float,
        V_hp_m3: float,
        beta: float,
    ) -> float:
        """dp_pump/dt = β/V * (Q_pump - Q_valves - Q_leak - Q_relief)."""
        Q_p = self.flow_m3s(epsilon)
        Q_leak = self.case_leak_flow_m3s(p_pump)
        dQ = Q_p - total_valve_flow_m3s - Q_leak
        return beta / (V_hp_m3 + _EPS) * dQ
