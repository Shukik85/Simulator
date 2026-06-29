from __future__ import annotations

import math

from hydrosim_v2.config.hydraulics import LSValveSectionConfig, FluidConfig
from hydrosim_v2.core.units import bar_to_Pa, LPM_to_m3s

_EPS = 1e-12


class LSValveSection:
    """One LS valve section: spool + pressure compensator.

    Supports extensions (u > 0) and retractions (u < 0).
    """

    def __init__(
        self,
        config: LSValveSectionConfig,
        fluid: FluidConfig,
    ):
        self.cfg = config
        self.rho = fluid.rho

        # Nominal conductance: K_v = C_d * A_max * sqrt(2/ρ)
        Q_nom_m3s = LPM_to_m3s(config.q_nom_lpm_at_dp)
        dp_nom = bar_to_Pa(config.delta_p_rated_bar)
        self.K_v = Q_nom_m3s / (math.sqrt(dp_nom) + _EPS)

    def spool_area_fraction(self, u: float) -> float:
        """Effective opening fraction [0, 1] for signal u ∈ [-1, 1]."""
        abs_u = abs(u)
        db = self.cfg.deadband
        if abs_u <= db:
            return 0.0
        frac = (abs_u - db) / (1.0 - db)
        return float(frac ** self.cfg.flow_exp)

    def flow_pa(self, u: float, p_P: float, p_A: float) -> float:
        """Flow from P → A (m³/s)."""
        if u <= _EPS:
            return 0.0
        dp_meter = min(p_P - p_A, bar_to_Pa(self.cfg.delta_p_rated_bar))
        if dp_meter <= _EPS:
            return 0.0
        frac = self.spool_area_fraction(u)
        return self.K_v * frac * math.sqrt(dp_meter)

    def flow_bt(self, u: float, p_B: float) -> float:
        """Flow from B → T (m³/s)."""
        if u <= _EPS or p_B <= _EPS:
            return 0.0
        frac = self.spool_area_fraction(u)
        return self.K_v * frac * math.sqrt(p_B)

    def flow_pb(self, u: float, p_P: float, p_B: float) -> float:
        """Flow from P → B (m³/s) during retraction."""
        if u >= -_EPS:
            return 0.0
        dp_meter = min(p_P - p_B, bar_to_Pa(self.cfg.delta_p_rated_bar))
        if dp_meter <= _EPS:
            return 0.0
        frac = self.spool_area_fraction(-u)
        return self.K_v * frac * math.sqrt(dp_meter)

    def flow_at(self, u: float, p_A: float) -> float:
        """Flow from A → T (m³/s) during retraction."""
        if u >= -_EPS or p_A <= _EPS:
            return 0.0
        frac = self.spool_area_fraction(-u)
        return self.K_v * frac * math.sqrt(p_A)

    def net_flow_A(self, u: float, p_P: float, p_A: float) -> float:
        """Net flow into chamber A. Positive = into cylinder."""
        return self.flow_pa(u, p_P, p_A) - self.flow_at(u, p_A)

    def net_flow_B(self, u: float, p_P: float, p_B: float) -> float:
        """Net flow into chamber B. Positive = into cylinder."""
        return self.flow_pb(u, p_P, p_B) - self.flow_bt(u, p_B)

    def supply_flow(self, u: float, p_P: float, p_A: float, p_B: float) -> float:
        """Flow drawn from pump for this section (m³/s)."""
        return self.flow_pa(u, p_P, p_A) + self.flow_pb(u, p_P, p_B)
