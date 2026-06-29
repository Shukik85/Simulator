from __future__ import annotations

from typing import Dict
import math

from hydrosim_v2.config import ExcavatorConfig
from hydrosim_v2.core.sim_state import SimState
from hydrosim_v2.kinematics.forward import forward_kinematics


class LoadModel:
    def __init__(self, cfg: ExcavatorConfig) -> None:
        self.cfg = cfg

    def external_cylinder_forces(self, state: SimState) -> Dict[str, float]:
        cfg = self.cfg
        mech = cfg.mechanics

        cyl_lengths = {name: mech.cylinders()[name].length_min_m + state.cyl_pos[name] for name in ("boom_cyl", "arm_cyl", "bucket_cyl")}

        pts = forward_kinematics(mech, cyl_lengths)

        g = 9.81

        m_bm = mech.boom_link.mass_kg
        m_am = mech.arm_link.mass_kg
        m_bk = mech.bucket_link.mass_kg

        l_bm = mech.boom_link.length_m
        l_am = mech.arm_link.length_m
        l_bk = mech.bucket_link.length_m

        com_bm_x = mech.boom_link.com_local[0]
        com_am_x = mech.arm_link.com_local[0]
        com_bk_x = mech.bucket_link.com_local[0]

        theta_boom = math.atan2(pts["boom_tip"][1] - pts["boom_joint"][1], pts["boom_tip"][0] - pts["boom_joint"][0])
        theta_arm = math.atan2(pts["arm_tip"][1] - pts["arm_joint"][1], pts["arm_tip"][0] - pts["arm_joint"][0])
        theta_bucket = math.atan2(pts["bucket_tip"][1] - pts["bucket_joint"][1], pts["bucket_tip"][0] - pts["bucket_joint"][0])

        m_g_bm = m_bm * g * com_bm_x * math.cos(theta_boom)
        m_g_am = m_am * g * (l_bm * math.cos(theta_boom) + com_am_x * math.cos(theta_arm))
        m_g_bk = m_bk * g * (l_bm * math.cos(theta_boom) + l_am * math.cos(theta_arm) + com_bk_x * math.cos(theta_bucket))

        return {
            "boom_cyl": -m_g_bm / 0.5,
            "arm_cyl": -m_g_am / 0.5,
            "bucket_cyl": -m_g_bk / 0.5,
        }
