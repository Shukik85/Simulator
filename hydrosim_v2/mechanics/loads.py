from __future__ import annotations

from typing import Dict
import math

from hydrosim_v2.config import ExcavatorConfig
from hydrosim_v2.core.sim_state import SimState
from hydrosim_v2.kinematics.forward import forward_kinematics


class LoadModel:
    """Compute external cylinder forces from gravity using virtual work.

    F_ext = dU/dL where U is gravitational potential energy.
    If extending the cylinder lifts the load (dU/dL > 0), gravity opposes
    extension, so f_ext > 0 in the equation: acc = (f_hyd - f_ext) / m.
    """

    _DELTA = 0.01

    def __init__(self, cfg: ExcavatorConfig) -> None:
        self.cfg = cfg

    def _potential_energy(self, mech, cyl_lengths: Dict[str, float],
                          prev_bucket_theta: float | None = None,
                          payload_kg: float = 0.0) -> float:
        pts = forward_kinematics(mech, cyl_lengths, prev_bucket_theta=prev_bucket_theta)
        g = 9.81

        U = 0.0

        # Boom
        boom_dx = pts["boom_tip"][0] - pts["boom_joint"][0]
        boom_dy = pts["boom_tip"][1] - pts["boom_joint"][1]
        theta_boom = math.atan2(boom_dy, boom_dx)
        com_bx = mech.boom_link.com_local[0]
        com_by = mech.boom_link.com_local[1]
        com_y = com_bx * math.sin(theta_boom) + com_by * math.cos(theta_boom)
        U += mech.boom_link.mass_kg * g * com_y

        # Arm
        arm_dx = pts["arm_tip"][0] - pts["arm_joint"][0]
        arm_dy = pts["arm_tip"][1] - pts["arm_joint"][1]
        theta_arm = math.atan2(arm_dy, arm_dx)
        com_ax = mech.arm_link.com_local[0]
        com_ay = mech.arm_link.com_local[1]
        com_y = com_ax * math.sin(theta_arm) + com_ay * math.cos(theta_arm)
        com_y += pts["arm_joint"][1]
        U += mech.arm_link.mass_kg * g * com_y

        # Bucket
        theta_bucket = pts["bucket_theta_rad"]
        com_bkx = mech.bucket_link.com_local[0]
        com_bky = mech.bucket_link.com_local[1]
        com_y = com_bkx * math.sin(theta_bucket) + com_bky * math.cos(theta_bucket)
        com_y += pts["bucket_joint"][1]
        U += (mech.bucket_link.mass_kg + payload_kg) * g * com_y

        return U

    def external_cylinder_forces(self, state: SimState, payload_kg: float = 0.0) -> Dict[str, float]:
        mech = self.cfg.mechanics
        cyl_names = ("boom_cyl", "arm_cyl", "bucket_cyl")

        cyl_lengths = {
            name: mech.cylinders()[name].length_min_m + state.cyl_pos[name]
            for name in cyl_names
        }

        base_pts = forward_kinematics(mech, cyl_lengths)
        prev_bucket_theta = base_pts.get("bucket_theta_rad")
        U0 = self._potential_energy(mech, cyl_lengths, prev_bucket_theta, payload_kg)

        forces = {}
        for name in cyl_names:
            cyl_perturbed = cyl_lengths.copy()
            cyl_perturbed[name] += self._DELTA
            try:
                U1 = self._potential_energy(mech, cyl_perturbed, prev_bucket_theta, payload_kg)
                forces[name] = (U1 - U0) / self._DELTA
            except Exception:
                forces[name] = 0.0

        return forces

    def potential_energy(self, state: SimState, payload_kg: float = 0.0) -> float:
        mech = self.cfg.mechanics
        cyl_names = ("boom_cyl", "arm_cyl", "bucket_cyl")
        cyl_lengths = {
            name: mech.cylinders()[name].length_min_m + state.cyl_pos[name]
            for name in cyl_names
        }
        base_pts = forward_kinematics(mech, cyl_lengths)
        prev_bucket_theta = base_pts.get("bucket_theta_rad")
        return self._potential_energy(mech, cyl_lengths, prev_bucket_theta, payload_kg)
