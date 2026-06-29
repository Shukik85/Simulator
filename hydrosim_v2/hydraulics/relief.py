"""Relief valve model."""

from __future__ import annotations

import numpy as np

from hydrosim_v2.config.hydraulics import ReliefValveConfig


class ReliefValve:
    def __init__(self, name: str, cfg: ReliefValveConfig) -> None:
        self.name = name
        self.cfg = cfg
        self.q_relief = 0.0

    def flow(self, p: float) -> float:
        cfg = self.cfg
        p_crack = cfg.crack_bar * 1e5
        p_max = cfg.max_bar * 1e5
        if p <= p_crack:
            self.q_relief = 0.0
        elif p >= p_max:
            self.q_relief = cfg.gain_m3_s_per_bar * (p_max - p_crack) / 1e5
        else:
            dp = p - p_crack
            self.q_relief = cfg.gain_m3_s_per_bar * (dp / 1e5)
        self.q_relief = max(self.q_relief, 0.0)
        return self.q_relief
