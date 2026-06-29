"""Inverse dynamics: gravitational torque compensation."""

from __future__ import annotations

from typing import Dict

import numpy as np

from typing import Dict

from hydrosim_v2.config import ExcavatorConfig
from hydrosim_v2.config.base import CylinderGeometry
from hydrosim_v2.kinematics.forward import forward_kinematics


class InverseDynamics:
    def __init__(self, cfg: ExcavatorConfig) -> None:
        self.cfg = cfg

    def compute_gravitational(
        self,
        cyl_lengths: Dict[str, float],
        geometries: Dict[str, CylinderGeometry],
    ) -> Dict[str, float]:
        cfg = self.cfg
        mech = cfg.mechanics
        g = 9.81

        pts = forward_kinematics(cfg.mechanics, cyl_lengths)
        boom_com = np.array(pts.get("boom_com", (0.0, 0.0)))
        arm_com = np.array(pts.get("arm_com", (0.0, 0.0)))
        bucket_com = np.array(pts.get("bucket_com", (0.0, 0.0)))

        m_bm = mech.boom_link.mass_kg
        m_am = mech.arm_link.mass_kg
        m_bk = mech.bucket_link.mass_kg

        G_bm = np.array([0.0, -m_bm * g])
        G_am = np.array([0.0, -m_am * g])
        G_bk = np.array([0.0, -m_bk * g])

        J_bm = np.zeros(2)
        J_am = np.zeros(2)
        J_bk = np.zeros(2)

        # Placeholder Jacobian transpose * weight
        tau_bm = float(J_bm[0])
        tau_am = float(J_am[0])
        tau_bk = float(J_bk[0])

        return {
            "boom_cyl": tau_bm,
            "arm_cyl": tau_am,
            "bucket_cyl": tau_bk,
        }
