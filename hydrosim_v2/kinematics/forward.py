from __future__ import annotations

from typing import Dict
import numpy as np

from hydrosim_v2.config import ExcavatorMechanicsConfig
from hydrosim_v2.kinematics.bucket_lever import solve_bucket

_EPS = 1e-12


def solve_link_angle(
    A: tuple[float, float],
    P_local: tuple[float, float],
    L_cyl: float,
    sign: int = 1,
) -> float:
    Aa = np.asarray(A, dtype=float)
    Pp = np.asarray(P_local, dtype=float)
    dA = float(np.linalg.norm(Aa))
    dP = float(np.linalg.norm(Pp))
    if dA < _EPS or dP < _EPS:
        return 0.0
    theta_A = float(np.arctan2(Aa[1], Aa[0]))
    theta_P = float(np.arctan2(Pp[1], Pp[0]))
    cos_alpha = (dA * dA + dP * dP - L_cyl * L_cyl) / (2.0 * dA * dP)
    cos_alpha = float(np.clip(cos_alpha, -1.0, 1.0))
    alpha = float(np.arccos(cos_alpha))
    return theta_A + sign * alpha - theta_P


def _rot2d(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def _vec2(x: float, y: float) -> tuple[float, float]:
    return (float(x), float(y))


def forward_kinematics(
    cfg: ExcavatorMechanicsConfig,
    cyl_lengths: Dict[str, float],
    prev_bucket_theta: float | None = None,
) -> Dict[str, tuple[float, float]]:
    base = (0.0, 0.0)
    base_arr = np.array([0.0, 0.0])

    A_boom = np.array(cfg.boom_cyl.base_mount.point_local, dtype=float)
    P_boom_local = np.array(cfg.boom_cyl.rod_mount.point_local, dtype=float)
    theta_boom = solve_link_angle(
        (float(A_boom[0]), float(A_boom[1])),
        (float(P_boom_local[0]), float(P_boom_local[1])),
        float(cyl_lengths["boom_cyl"]),
        sign=1,
    )
    R_boom = _rot2d(theta_boom)
    boom_joint = base_arr.copy()
    boom_tip = boom_joint + R_boom @ np.array([cfg.boom_link.length_m, 0.0])
    A_boom_global = _vec2(float(A_boom[0]), float(A_boom[1]))
    P_boom_global = _vec2(*(boom_joint + R_boom @ P_boom_local))

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
        sign=-1,
    )
    R_arm = _rot2d(theta_arm)
    arm_tip = arm_joint + R_arm @ np.array([cfg.arm_link.length_m, 0.0])
    P_arm_global = _vec2(*(arm_joint + R_arm @ P_arm_local))

    sol = solve_bucket(
        cfg.bucket_lever,
        float(cyl_lengths["bucket_cyl"]),
        theta_arm,
        _vec2(float(arm_joint[0]), float(arm_joint[1])),
        prev_theta=prev_bucket_theta,
    )
    if sol is None:
        raise RuntimeError(f"Cannot solve bucket kinematics for L_cyl={cyl_lengths['bucket_cyl']}")

    return {
        "base": base,
        "boom_joint": base,
        "boom_tip": _vec2(float(boom_tip[0]), float(boom_tip[1])),
        "arm_joint": _vec2(float(arm_joint[0]), float(arm_joint[1])),
        "arm_tip": _vec2(float(arm_tip[0]), float(arm_tip[1])),
        "bucket_joint": sol["D"],
        "bucket_tip": sol["E"],
        "bucket_tip_cutting_edge": sol["bucket_tip"],
        "bucket_com": sol["com"],
        "bucket_theta_rad": sol["theta_rad"],
        "A_boom": A_boom_global,
        "P_boom": P_boom_global,
        "A_arm": _vec2(float(A_arm_absolute[0]), float(A_arm_absolute[1])),
        "P_arm": P_arm_global,
    }
