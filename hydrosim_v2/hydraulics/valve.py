"""LS valve section model."""

from __future__ import annotations

import numpy as np

from hydrosim_v2.config.hydraulics import LSValveSectionConfig


class LSValveSection:
    def __init__(self, name: str, cfg: LSValveSectionConfig) -> None:
        self.name = name
        self.cfg = cfg
        self.spool = 0.0
        self.A_p = 0.0
        self.A_t = 0.0
        self.q_p = 0.0
        self.q_t = 0.0

    @property
    def q_nom_m3_s(self) -> float:
        return self.cfg.q_nom_lpm_at_dp / 60000.0

    @property
    def k_v(self) -> float:
        dpp = self.cfg.delta_p_rated_bar * 1e5
        return self.q_nom_m3_s / np.sqrt(dpp)

    def _area(self, s: float) -> float:
        db = self.cfg.deadband
        if abs(s) < db:
            return 0.0
        s_eff = (abs(s) - db) / (1.0 - db)
        s_eff = float(np.clip(s_eff, 0.0, 1.0))
        return s_eff ** self.cfg.flow_exp

    def step(self, spool_cmd: float, p_p: float, p_a: float, p_b: float, rho: float = 850.0) -> None:
        self.spool = float(np.clip(spool_cmd, -1.0, 1.0))
        A = self._area(self.spool)
        p_t = 0.0
        kv = self.k_v

        if self.spool >= 0:
            self.A_p = A
            self.A_t = A
            dp_pa = p_p - p_a
            self.q_p = 0.0 if abs(dp_pa) < 1e3 else kv * A * float(np.sign(dp_pa)) * float(np.sqrt(abs(dp_pa) / rho))
            dp_bt = p_b - p_t
            self.q_t = 0.0 if abs(dp_bt) < 1e3 else kv * A * float(np.sign(dp_bt)) * float(np.sqrt(abs(dp_bt) / rho))
        else:
            self.A_p = A
            self.A_t = A
            dp_pb = p_p - p_b
            self.q_p = 0.0 if abs(dp_pb) < 1e3 else kv * A * float(np.sign(dp_pb)) * float(np.sqrt(abs(dp_pb) / rho))
            dp_at = p_a - p_t
            self.q_t = 0.0 if abs(dp_at) < 1e3 else kv * A * float(np.sign(dp_at)) * float(np.sqrt(abs(dp_at) / rho))

    @property
    def q_a(self) -> float:
        if self.spool >= 0:
            return self.q_p
        else:
            return -self.q_t

    @property
    def q_b(self) -> float:
        if self.spool >= 0:
            return -self.q_t
        else:
            return self.q_p
