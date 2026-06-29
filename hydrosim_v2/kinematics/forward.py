from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from hydrosim_v2.config import ExcavatorMechanicsConfig
from hydrosim_v2.core.types import Vec2
from hydrosim_v2.kinematics.bucket_lever import solve_bucket

LOG = logging.getLogger(__name__)
_EPS = 1e-12


def solve_link_angle(
    A: Vec2,
    P_local: Vec2,
    L_cyl: float,
    sign: int = 1,
) -> float:
    """Безопасно решает child-link angle θ such that |R(θ) @ P_local - A| = L_cyl.
    Возвращает 0.0 при невозможности построения треугольника.
    """
    Aa = np.asarray(A, dtype=float)
    Pp = np.asarray(P_local, dtype=float)
    dA = float(np.linalg.norm(Aa))
    dP = float(np.linalg.norm(Pp))
    if dA < _EPS or dP < _EPS:
        return 0.0
    # Проверка существования треугольника
    if (dA + dP < L_cyl - _EPS) or (abs(dA - dP) > L_cyl + _EPS):
        return 0.0
    cos_alpha = (dA * dA + dP * dP - L_cyl * L_cyl) / (2.0 * dA * dP)
    cos_alpha = float(np.clip(cos_alpha, -1.0, 1.0))
    phi = float(np.arctan2(
        Pp[0] * Aa[1] - Pp[1] * Aa[0],
        Pp[0] * Aa[0] + Pp[1] * Aa[1],
    ))
    try:
        acos_val = float(np.arccos(cos_alpha))
    except ValueError:
        acos_val = 0.0
    theta = phi + sign * acos_val
    # Привести угол к диапазону [-pi, pi]
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    return theta



def _rot2d(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def forward_kinematics(
    cfg: ExcavatorMechanicsConfig,
    cyl_lengths: Dict[str, float],
) -> Dict[str, Vec2]:
    """Compute all joint positions from cylinder lengths.

    Returns dict with keys:
      base, boom_joint, boom_tip, arm_joint, arm_tip,
      bucket_joint, bucket_tip, bucket_tip_cutting_edge, bucket_com,
      A_boom, P_boom, A_arm, P_arm
    """
    base = np.array([0.0, 0.0])

    # --- Boom ---
    A_boom = np.array(cfg.boom_cyl.base_mount.point_local, dtype=float)
    P_boom_local = np.array(cfg.boom_cyl.rod_mount.point_local, dtype=float)
    theta_boom = solve_link_angle(
        (float(A_boom[0]), float(A_boom[1])),
        (float(P_boom_local[0]), float(P_boom_local[1])),
        float(cyl_lengths["boom_cyl"]),
        sign=1,
    )
    R_boom = _rot2d(theta_boom)
    boom_joint = base.copy()
    boom_tip = boom_joint + R_boom @ np.array([cfg.boom_link.length_m, 0.0])
    A_boom_global = (float(A_boom[0]), float(A_boom[1]))
    P_boom_global = tuple(float(x) for x in (boom_joint + R_boom @ P_boom_local))

    # --- Arm ── base mount on rotated boom, rod mount on arm ---
    A_arm_absolute = boom_joint + R_boom @ np.array(
        cfg.arm_cyl.base_mount.point_local, dtype=float
    )
    arm_joint = boom_tip.copy()
    A_prime = A_arm_absolute - arm_joint
    P_arm_local = np.array(cfg.arm_cyl.rod_mount.point_local, dtype=float)
    theta_arm = solve_link_angle(
        (float(A_prime[0]), float(A_prime[1])),
        (float(P_arm_local[0]), float(P_arm_local[1])),
        float(cyl_lengths["arm_cyl"]),
        sign=1,
    )
    R_arm = _rot2d(theta_arm)
    arm_tip = arm_joint + R_arm @ np.array([cfg.arm_link.length_m, 0.0])
    P_arm_global = tuple(float(x) for x in (arm_joint + R_arm @ P_arm_local))

    # --- Bucket ---
    sol = solve_bucket(
        cfg.bucket_lever,
        float(cyl_lengths["bucket_cyl"]),
        theta_arm,
        (float(arm_joint[0]), float(arm_joint[1])),
    )
    if sol is None:
        msg = f"Cannot solve bucket kinematics for L_cyl={cyl_lengths['bucket_cyl']}"
        LOG.error(msg)
        raise RuntimeError(msg)

    LOG.info(
        "FK: θ_boom=%.1f° θ_arm=%.1f° θ_bucket=%.1f°",
        np.degrees(theta_boom), np.degrees(theta_arm), np.degrees(sol.theta_rad),
    )

    return {
        "base": (0.0, 0.0),
        "boom_joint": (0.0, 0.0),
        "boom_tip": (float(boom_tip[0]), float(boom_tip[1])),
        "arm_joint": (float(arm_joint[0]), float(arm_joint[1])),
        "arm_tip": (float(arm_tip[0]), float(arm_tip[1])),
        "bucket_joint": sol.D,
        "bucket_tip": sol.E,
        "bucket_tip_cutting_edge": sol.bucket_tip,
        "bucket_com": sol.com,
        "A_boom": A_boom_global,
        "P_boom": P_boom_global,
        "A_arm": (float(A_arm_absolute[0]), float(A_arm_absolute[1])),
        "P_arm": P_arm_global,
    }
