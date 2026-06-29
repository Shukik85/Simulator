from __future__ import annotations

from typing import Dict
import numpy as np

from hydrosim_v2.config import ExcavatorMechanicsConfig
from hydrosim_v2.kinematics.forward import forward_kinematics


def backward_static(
    cfg: ExcavatorMechanicsConfig,
    cyl_lengths: Dict[str, float],
    ee_force: tuple[float, float],
    ee_point: str = "bucket_tip",
    delta: float = 1e-6,
) -> Dict[str, float]:
    cyl_names = list(cyl_lengths.keys())
    n_cyl = len(cyl_names)
    J = np.zeros((2, n_cyl))

    for i, name in enumerate(cyl_names):
        L_plus = cyl_lengths.copy()
        L_minus = cyl_lengths.copy()
        L_plus[name] += delta
        L_minus[name] -= delta
        pts_plus = forward_kinematics(cfg, L_plus)
        pts_minus = forward_kinematics(cfg, L_minus)
        dx = (np.array(pts_plus[ee_point]) - np.array(pts_minus[ee_point])) / (2.0 * delta)
        J[:, i] = dx

    tau = J.T @ np.array(ee_force)
    return {name: float(tau[i]) for i, name in enumerate(cyl_names)}
