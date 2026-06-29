from __future__ import annotations

from hydrosim_v2.config.hydraulics import ReliefValveConfig
from hydrosim_v2.core.units import bar_to_Pa

_EPS = 1e-12


class ReliefValve:
    """Simple pressure relief valve model."""

    def __init__(self, config: ReliefValveConfig):
        self.cfg = config
        self.crack_pa = bar_to_Pa(config.crack_bar)
        self.max_pa = bar_to_Pa(config.max_bar)
        self.gain_m3s_per_pa = config.gain_m3_s_per_bar / 1e5  # per bar → per Pa

    def flow_m3s(self, p_pump: float) -> float:
        """Relief flow (m³/s) when pump pressure exceeds crack."""
        if p_pump <= self.crack_pa:
            return 0.0
        excess = min(p_pump - self.crack_pa, self.max_pa - self.crack_pa)
        return excess * self.gain_m3s_per_pa
