"""Ковшевой рычажный механизм (плоская кинематика 2D).

Это перенос и упрощение логики из корневого `lever_system.py` внутрь пакета `hydrosim`.
Цель: убрать внешние зависимости (SciPy) и дать API с выбором ветки, устойчивый
к движению родительского звена.

Ветка (branch) задаётся знаком `desired_a_side`:
- +1: точка A находится "выше" линии OG (положительный cross(OG, OA))
- -1: точка A "ниже" линии OG

Публичный API:
- solve_angle(cyl_length_m) -> (theta_rad, dtheta_dl)
- solve_angle_with_branch(cyl_length_m, branch_prev=...) -> (theta_rad, dtheta_dl, branch)

Примечание:
- Угол theta соответствует углу BV (как в исходной модели).
- Решение ограничений выполняется damped-Newton без SciPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

Vec2 = NDArray[np.float64]
Vec4 = NDArray[np.float64]
BranchSign = Literal[-1, 1]


def v2(x: float, y: float) -> Vec2:
    return np.array([x, y], dtype=np.float64)


def cross2_z(a: Vec2, b: Vec2) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def wrap_angle_pi(a: float) -> float:
    """Приведение угла к диапазону (-π, π] с канонизацией -π -> +π."""

    out = float((a + np.pi) % (2.0 * np.pi) - np.pi)
    if np.isclose(out, -np.pi):
        return float(np.pi)
    return out


@dataclass(frozen=True, slots=True)
class FixedPoints:
    O: Vec2
    V: Vec2
    G: Vec2


@dataclass(frozen=True, slots=True)
class LinkLengths:
    AB: float
    BV: float
    GA: float
    OA_min: float
    OA_max: float


@dataclass(frozen=True, slots=True)
class KinematicState:
    l_OA: float
    A: Vec2
    B: Vec2
    fixed: FixedPoints
    lengths: LinkLengths
    residual_norm: float
    J_constraint: Optional[NDArray[np.float64]] = None


class KinematicsError(RuntimeError):
    pass


class BucketLeverGeometry:
    """Геометрический решатель рычажной системы ковша (без SciPy)."""

    def __init__(
        self,
        *,
        fixed_points_m: dict[str, tuple[float, float]],
        A0_m: tuple[float, float],
        B0_m: tuple[float, float],
        OA_min_m: float,
        OA_max_m: float,
        desired_a_side: BranchSign = 1,
    ) -> None:
        self.fixed = FixedPoints(
            O=v2(float(fixed_points_m["O"][0]), float(fixed_points_m["O"][1])),
            V=v2(float(fixed_points_m["V"][0]), float(fixed_points_m["V"][1])),
            G=v2(float(fixed_points_m["G"][0]), float(fixed_points_m["G"][1])),
        )
        A0 = v2(float(A0_m[0]), float(A0_m[1]))
        B0 = v2(float(B0_m[0]), float(B0_m[1]))

        self.lengths = LinkLengths(
            AB=float(np.linalg.norm(B0 - A0)),
            BV=float(np.linalg.norm(B0 - self.fixed.V)),
            GA=float(np.linalg.norm(A0 - self.fixed.G)),
            OA_min=float(OA_min_m),
            OA_max=float(OA_max_m),
        )

        self.desired_a_side: BranchSign = 1 if desired_a_side >= 0 else -1
        self._last_x: Optional[Vec4] = None

    def a_side_sign(self, A: Vec2) -> int:
        og = self.fixed.G - self.fixed.O
        oa = A - self.fixed.O
        cz = cross2_z(og, oa)
        return 1 if cz > 1e-14 else (-1 if cz < -1e-14 else 0)

    def _constraints(self, x: Vec4, l_OA: float) -> Vec4:
        """Ограничения в форме квадратов расстояний (4 уравнения / 4 неизвестных)."""

        A, B = x[:2], x[2:]
        O, V, G = self.fixed.O, self.fixed.V, self.fixed.G
        L = self.lengths
        return np.array(
            [
                float(np.dot(A - O, A - O) - l_OA * l_OA),
                float(np.dot(B - A, B - A) - L.AB * L.AB),
                float(np.dot(B - V, B - V) - L.BV * L.BV),
                float(np.dot(A - G, A - G) - L.GA * L.GA),
            ],
            dtype=np.float64,
        )

    def _jacobian_constraints(self, x: Vec4) -> NDArray[np.float64]:
        A = x[:2]
        B = x[2:]
        O = self.fixed.O
        V = self.fixed.V
        G = self.fixed.G

        J = np.zeros((4, 4), dtype=np.float64)
        J[0, :2] = 2.0 * (A - O)
        J[1, :2] = -2.0 * (B - A)
        J[1, 2:] = 2.0 * (B - A)
        J[2, 2:] = 2.0 * (B - V)
        J[3, :2] = 2.0 * (A - G)
        return J

    def _initial_guesses(self, l_OA: float) -> list[Vec4]:
        O, V = self.fixed.O, self.fixed.V
        L = self.lengths

        guesses: list[Vec4] = []
        if self._last_x is not None:
            guesses.append(self._last_x)

        dir_ov = V - O
        n = float(np.linalg.norm(dir_ov))
        if n > 1e-14:
            dir_ov = dir_ov / n
            rot90 = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)

            A1 = O + l_OA * dir_ov
            B1 = V - L.BV * dir_ov
            guesses.append(np.array([A1[0], A1[1], B1[0], B1[1]], dtype=np.float64))

            dir_perp = rot90 @ dir_ov
            A2 = O + l_OA * dir_perp
            B2 = V - L.BV * dir_perp
            guesses.append(np.array([A2[0], A2[1], B2[0], B2[1]], dtype=np.float64))

            A3 = O - l_OA * dir_perp
            B3 = V + L.BV * dir_ov
            guesses.append(np.array([A3[0], A3[1], B3[0], B3[1]], dtype=np.float64))

        return guesses

    def solve(self, l_OA_m: float, *, max_residual: float = 1e-6) -> KinematicState:
        l_OA = float(l_OA_m)
        if not (self.lengths.OA_min <= l_OA <= self.lengths.OA_max):
            raise ValueError(f"l_OA={l_OA} m out of range")

        best: Optional[Vec4] = None
        best_norm = float("inf")
        best_J: Optional[NDArray[np.float64]] = None

        for x0 in self._initial_guesses(l_OA):
            x = x0.copy()
            ok = False
            for _ in range(60):
                f = self._constraints(x, l_OA)
                f_norm = float(np.linalg.norm(f))
                if f_norm < max_residual:
                    ok = True
                    break

                J = self._jacobian_constraints(x)
                try:
                    dx = np.linalg.solve(J, -f)
                except np.linalg.LinAlgError:
                    break

                # Damped step (simple backtracking)
                alpha = 1.0
                for _ls in range(12):
                    x_new = x + alpha * dx
                    f_new = self._constraints(x_new, l_OA)
                    if float(np.linalg.norm(f_new)) < f_norm:
                        x = x_new
                        break
                    alpha *= 0.5
                else:
                    break

            if not ok:
                continue

            A = x[:2]
            if self.a_side_sign(A) != int(self.desired_a_side):
                continue

            f = self._constraints(x, l_OA)
            f_norm = float(np.linalg.norm(f))
            if f_norm < best_norm:
                best_norm = f_norm
                best = x.copy()
                best_J = self._jacobian_constraints(x)

        if best is None:
            raise KinematicsError("No solution on desired branch")

        A = best[:2].copy()
        B = best[2:].copy()
        self._last_x = best

        return KinematicState(
            l_OA=float(l_OA),
            A=A,
            B=B,
            fixed=self.fixed,
            lengths=self.lengths,
            residual_norm=float(best_norm),
            J_constraint=best_J,
        )

    def theta_BV(self, state: KinematicState) -> float:
        v = state.fixed.V - state.B
        return float(np.arctan2(v[1], v[0]))


class BucketLeverMechanism:
    """Упрощённый интерфейс ковшевого рычага для интеграции в `kinematics`.

    Важно: ветка выбирается через `branch_prev` (A-side), что инвариантно
    к глобальному повороту/движению родительского звена.
    """

    def __init__(
        self,
        *,
        fixed_points_m: dict[str, tuple[float, float]],
        A0_m: tuple[float, float],
        B0_m: tuple[float, float],
        OA_min_m: float,
        OA_max_m: float,
        default_branch: BranchSign = 1,
    ) -> None:
        self._fixed_points_m = fixed_points_m
        self._A0_m = A0_m
        self._B0_m = B0_m
        self._OA_min_m = float(OA_min_m)
        self._OA_max_m = float(OA_max_m)
        self._default_branch: BranchSign = 1 if default_branch >= 0 else -1

        self._geom = BucketLeverGeometry(
            fixed_points_m=fixed_points_m,
            A0_m=A0_m,
            B0_m=B0_m,
            OA_min_m=self._OA_min_m,
            OA_max_m=self._OA_max_m,
            desired_a_side=self._default_branch,
        )

    def solve_angle(self, cyl_length_m: float) -> Tuple[float, float]:
        theta, dtheta_dl, _branch = self.solve_angle_with_branch(float(cyl_length_m))
        return float(theta), float(dtheta_dl)

    def solve_angle_with_branch(
        self,
        cyl_length_m: float,
        *,
        branch_prev: BranchSign | None = None,
    ) -> Tuple[float, float, BranchSign]:
        branch: BranchSign = self._default_branch if branch_prev is None else (1 if branch_prev >= 0 else -1)

        # Если ветка изменилась, пересоздаём геометрию с нужным ограничением.
        # Это дешево и гарантирует корректный side-предикат.
        if branch != self._geom.desired_a_side:
            self._geom = BucketLeverGeometry(
                fixed_points_m=self._fixed_points_m,
                A0_m=self._A0_m,
                B0_m=self._B0_m,
                OA_min_m=self._OA_min_m,
                OA_max_m=self._OA_max_m,
                desired_a_side=branch,
            )

        state = self._geom.solve(float(cyl_length_m))
        theta = wrap_angle_pi(self._geom.theta_BV(state))

        # dtheta/dl через неявное дифференцирование: J_x * dx/dl = -J_l
        J_x = state.J_constraint
        if J_x is None:
            dtheta_dl = 0.0
        else:
            J_l = np.array([-2.0 * state.l_OA, 0.0, 0.0, 0.0], dtype=np.float64)
            try:
                dx_dl = np.linalg.solve(J_x, -J_l)
            except np.linalg.LinAlgError:
                dx_dl = np.zeros(4, dtype=np.float64)

            V = state.fixed.V
            B = state.B
            r = V - B
            r_sq = float(r[0] * r[0] + r[1] * r[1])
            if r_sq < 1e-12:
                dtheta_dl = 0.0
            else:
                dtheta_dB = np.array([-r[1] / r_sq, r[0] / r_sq], dtype=np.float64)
                dtheta_dl = float(np.dot(dtheta_dB, dx_dl[2:4]))

        return float(theta), float(dtheta_dl), branch
