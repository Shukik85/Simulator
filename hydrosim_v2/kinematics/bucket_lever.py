from __future__ import annotations

from typing import Optional
import numpy as np

from hydrosim_v2.config.base import BucketLeverMechanismParams

_EPS = 1e-12


def _rot2d(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def _solve_triangle(
    base: np.ndarray, tip: np.ndarray,
    L1: float, L2: float,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    vec = tip - base
    d = np.linalg.norm(vec)
    if d < _EPS or d > L1 + L2 + _EPS or d < abs(L1 - L2) - _EPS:
        return None, None
    cos_a = (L1**2 + d**2 - L2**2) / (2 * L1 * d)
    cos_a = float(np.clip(cos_a, -1.0, 1.0))
    a = np.arccos(cos_a)
    angle = np.arctan2(float(vec[1]), float(vec[0]))
    P1 = base + L1 * np.array([np.cos(angle + a), np.sin(angle + a)])
    P2 = base + L1 * np.array([np.cos(angle - a), np.sin(angle - a)])
    return P1, P2


def bucket_cylinder_length(
    params: BucketLeverMechanismParams,
    theta_bucket: float,
    arm_angle_rad: float,
    arm_pivot_xy: tuple[float, float],
) -> Optional[float]:
    R_arm = _rot2d(arm_angle_rad)
    pivot = np.array(arm_pivot_xy)
    D = pivot + R_arm @ np.array(params.pivot_D)
    C = pivot + R_arm @ np.array(params.pivot_C)
    A = pivot + R_arm @ np.array(params.anchor_A)

    R_bucket = _rot2d(theta_bucket)
    E = D + R_bucket @ np.array(params.E_local)

    P1, P2 = _solve_triangle(np.array(C), np.array(E),
                              params.lever_length_m, params.rod_length_m)
    candidates = [p for p in (P1, P2) if p is not None]
    if not candidates:
        return None

    P_chosen = max(candidates, key=lambda p: float(p[1]))
    return float(np.linalg.norm(P_chosen - A))


def solve_bucket(
    params: BucketLeverMechanismParams,
    L_cyl: float,
    arm_angle_rad: float,
    arm_pivot_xy: tuple[float, float],
) -> Optional[dict[str, tuple[float, float]]]:
    L_cyl = float(L_cyl)

    R_arm = _rot2d(arm_angle_rad)
    pivot = np.array(arm_pivot_xy)
    A = pivot + R_arm @ np.array(params.anchor_A)
    C = pivot + R_arm @ np.array(params.pivot_C)
    D = pivot + R_arm @ np.array(params.pivot_D)

    theta_min, theta_max = -2 * np.pi / 3, 5 * np.pi / 6
    _prev_P: list[Optional[np.ndarray]] = [None]

    def residual(theta: float) -> float:
        R_bucket = _rot2d(theta)
        E = D + R_bucket @ np.array(params.E_local)
        P1, P2 = _solve_triangle(C, E, params.lever_length_m, params.rod_length_m)
        candidates = [p for p in (P1, P2) if p is not None]
        if not candidates:
            return 1e9
        if _prev_P[0] is not None:
            Pchosen = min(candidates, key=lambda p: float(np.linalg.norm(p - _prev_P[0])))
        else:
            Pchosen = candidates[0]
        _prev_P[0] = Pchosen.copy()
        return float(np.linalg.norm(Pchosen - A)) - L_cyl

    n_samples = 200
    thetas = np.linspace(theta_min, theta_max, n_samples)
    residuals = np.array([residual(t) for t in thetas])

    bracket: Optional[tuple[float, float]] = None
    for i in range(n_samples - 1):
        if residuals[i] * residuals[i + 1] < 0:
            bracket = (float(thetas[i]), float(thetas[i + 1]))
            break

    if bracket is None:
        idx = int(np.argmin(np.abs(residuals)))
        theta_best = float(thetas[idx])
        P_best = _prev_P[0]
        if P_best is None:
            return None
    else:
        a, b = bracket
        for _ in range(80):
            m = (a + b) * 0.5
            fm = residual(m)
            if fm == 0:
                a = b = m
                break
            fa = residual(a)
            if fa * fm < 0:
                b = m
            else:
                a = m
        theta_best = (a + b) * 0.5
        P_best = _prev_P[0]
        if P_best is None:
            return None

    R_bucket = _rot2d(theta_best)
    E = D + R_bucket @ np.array(params.E_local)
    com = D + R_bucket @ np.array(params.E_local)
    tip = D + R_bucket @ np.array(params.bucket_tip_local)

    return {
        "theta_rad": theta_best,
        "A": (float(A[0]), float(A[1])),
        "C": (float(C[0]), float(C[1])),
        "D": (float(D[0]), float(D[1])),
        "P": (float(P_best[0]), float(P_best[1])),
        "E": (float(E[0]), float(E[1])),
        "bucket_tip": (float(tip[0]), float(tip[1])),
        "com": (float(com[0]), float(com[1])),
    }
