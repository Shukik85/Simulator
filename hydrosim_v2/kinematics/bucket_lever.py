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


def _pick_branch(
    C: np.ndarray, E: np.ndarray, A: np.ndarray,
    P1: Optional[np.ndarray], P2: Optional[np.ndarray],
    L_cyl: float,
) -> Optional[np.ndarray]:
    candidates = [p for p in (P1, P2) if p is not None]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return min(candidates, key=lambda p: float(abs(np.linalg.norm(p - A) - L_cyl)))


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

    CE = E - C
    cross1 = float(CE[0] * (P1[1] - C[1]) - CE[1] * (P1[0] - C[0]))
    P_chosen = P1 if cross1 > 0 else P2
    return float(np.linalg.norm(P_chosen - A))


def _bisect(residual, a: float, b: float, n_iter: int = 80) -> float:
    for _ in range(n_iter):
        m = (a + b) * 0.5
        fm = residual(m)
        if fm == 0:
            return m
        fa = residual(a)
        if fa * fm < 0:
            b = m
        else:
            a = m
    return (a + b) * 0.5


def solve_bucket(
    params: BucketLeverMechanismParams,
    L_cyl: float,
    arm_angle_rad: float,
    arm_pivot_xy: tuple[float, float],
    prev_theta: Optional[float] = None,
) -> Optional[dict[str, tuple[float, float]]]:
    L_cyl = float(L_cyl)

    R_arm = _rot2d(arm_angle_rad)
    pivot = np.array(arm_pivot_xy)
    A = pivot + R_arm @ np.array(params.anchor_A)
    C = pivot + R_arm @ np.array(params.pivot_C)
    D = pivot + R_arm @ np.array(params.pivot_D)

    theta_min_full, theta_max_full = -2 * np.pi / 3, 5 * np.pi / 6

    if prev_theta is not None:
        margin = 0.8
        theta_min = max(theta_min_full, prev_theta - margin)
        theta_max = min(theta_max_full, prev_theta + margin)
    else:
        theta_min, theta_max = theta_min_full, theta_max_full

    def residual(theta: float) -> float:
        R_bucket = _rot2d(theta)
        E = D + R_bucket @ np.array(params.E_local)
        P1, P2 = _solve_triangle(C, E, params.lever_length_m, params.rod_length_m)
        if P1 is None and P2 is None:
            return 1e9
        if P1 is not None and P2 is not None:
            CE = E - C
            cross = float(CE[0] * (P1[1] - C[1]) - CE[1] * (P1[0] - C[0]))
            P = P1 if cross > 0 else P2
        else:
            P = P1 if P1 is not None else P2
        return float(np.linalg.norm(P - A)) - L_cyl

    n_samples = 200
    thetas = np.linspace(theta_min, theta_max, n_samples)
    residuals = np.array([residual(t) for t in thetas])

    candidates_theta: list[float] = []

    for i in range(n_samples - 1):
        if residuals[i] * residuals[i + 1] < 0:
            candidates_theta.append(_bisect(residual, float(thetas[i]), float(thetas[i + 1])))
        elif abs(residuals[i]) < 1e-6:
            candidates_theta.append(float(thetas[i]))

    if abs(residuals[-1]) < 1e-6:
        candidates_theta.append(float(thetas[-1]))

    if candidates_theta:
        if prev_theta is not None:
            best_theta = min(candidates_theta, key=lambda t: abs(t - prev_theta))
        else:
            best_theta = min(candidates_theta, key=lambda t: abs(residual(t)))
    else:
        idx = int(np.argmin(np.abs(residuals)))
        best_theta = float(thetas[idx])

    R_bucket = _rot2d(best_theta)
    E = D + R_bucket @ np.array(params.E_local)
    P1, P2 = _solve_triangle(C, E, params.lever_length_m, params.rod_length_m)
    P_best = _pick_branch(C, E, A, P1, P2, L_cyl)
    if P_best is None:
        return None

    tip = D + R_bucket @ np.array(params.bucket_tip_local)

    return {
        "theta_rad": best_theta,
        "A": (float(A[0]), float(A[1])),
        "C": (float(C[0]), float(C[1])),
        "D": (float(D[0]), float(D[1])),
        "P": (float(P_best[0]), float(P_best[1])),
        "E": (float(E[0]), float(E[1])),
        "bucket_tip": (float(tip[0]), float(tip[1])),
        "com": (float(E[0]), float(E[1])),
    }
