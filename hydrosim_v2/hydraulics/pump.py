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
        self.vol_eff = 1.0
        self.mech_eff = 1.0

    @property
    def max_flow_m3_s(self) -> float:
        cfg = self.cfg
        return cfg.max_displacement_cc_rev * 1e-6 * cfg.max_speed_rpm / 60.0

    @property
    def displacement_m3(self) -> float:
        return self.cfg.max_displacement_cc_rev * 1e-6 * self.swash

    @property
    def max_hyd_power_w(self) -> float:
        return self.cfg.power_hp * 745.699872 * self.cfg.mech_eff_0

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
        target_dp = max(target_margin, 3.0e5)

        # LS error: actual margin vs target
        actual_margin = p_out - p_ls
        error = target_dp - actual_margin

        # Swash PI: integrate error → swash (with anti-windup)
        tau = cfg.response_time_s
        alpha = dt / max(tau, 1e-6)
        alpha = min(alpha, 1.0)
        if error > 0 or self.dp_ls > 0:
            self.dp_ls += alpha * (error - self.dp_ls)

        swash_target = float(np.clip(self.dp_ls / target_dp, 0.0, 1.0))
        self.swash = swash_target

        speed_rev_s = pump_speed_rpm / 60.0
        D_max = cfg.max_displacement_cc_rev * 1e-6
        ideal_flow = D_max * self.swash * speed_rev_s
        vol_eff = cfg.vol_eff_0 - cfg.vol_eff_kp * (p_out / 1e7)
        vol_eff = float(np.clip(vol_eff, 0.85, 0.98))
        q_ideal = ideal_flow * vol_eff
        mech_eff = cfg.mech_eff_0 - cfg.mech_eff_kp * (p_out / 1e7)
        mech_eff = float(np.clip(mech_eff, 0.80, 0.95))
        self.vol_eff = vol_eff
        self.mech_eff = mech_eff
        self.mech_loss = (1.0 / max(mech_eff, 0.5) - 1.0) * q_ideal * p_out
        self.q_out = q_ideal
        return q_ideal
