"""Load-sensing pump model."""

from __future__ import annotations

import numpy as np

from hydrosim_v2.config.hydraulics import LSPumpConfig


class LSPump:
    def __init__(self, cfg: LSPumpConfig) -> None:
        self.cfg = cfg
        self.swash = 0.0
        self.dp_ls = 0.0
        self.ls_pressure = 0.0
        self.q_out = 0.0
        self.mech_loss = 0.0

    @property
    def max_flow_m3_s(self) -> float:
        cfg = self.cfg
        return cfg.max_displacement_cc_rev * 1e-6 * cfg.max_speed_rpm / 60.0

    @property
    def displacement_m3(self) -> float:
        return self.cfg.max_displacement_cc_rev * 1e-6 * self.swash

    def step(
        self,
        pump_speed_rpm: float,
        p_out: float,
        p_ls: float,
        dt: float,
    ) -> float:
        cfg = self.cfg
        p_ls = max(p_ls, 0.0)

        target_margin = cfg.margin_standby_bar * 1e5
        if self.q_out > 0.05 * self.max_flow_m3_s:
            target_margin += 0.0
        target_dp = max(target_margin, 3.0e5)

        target = p_ls + target_dp
        error = target - p_out
        tau = cfg.response_time_s
        alpha = dt / tau if tau > 0 else 1.0
        alpha = min(alpha, 1.0)
        self.dp_ls += alpha * (error - self.dp_ls)

        speed_rad_s = pump_speed_rpm * (2.0 * np.pi / 60.0)
        D_target = self.q_out / speed_rad_s if speed_rad_s > 1.0 else 0.0
        D_max = self.cfg.max_displacement_cc_rev * 1e-6
        self.swash = float(np.clip(D_target / D_max, 0.0, 1.0))

        ideal_flow = D_max * self.swash * speed_rad_s
        vol_eff = cfg.vol_eff_0 - cfg.vol_eff_kp * (p_out / 1e7)
        vol_eff = float(np.clip(vol_eff, 0.85, 0.98))
        q_ideal = ideal_flow * vol_eff
        mech_eff = cfg.mech_eff_0 - cfg.mech_eff_kp * (p_out / 1e7)
        mech_eff = float(np.clip(mech_eff, 0.80, 0.95))
        self.mech_loss = (1.0 - mech_eff) * q_ideal * p_out
        self.q_out = q_ideal
        return q_ideal
