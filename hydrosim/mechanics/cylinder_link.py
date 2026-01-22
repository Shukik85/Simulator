"""Кинематическая и динамическая связь гидроцилиндра и звена.

Решает:
- по длине цилиндра -> угол звена и его производная dθ/dl;
- по силе в цилиндре -> момент на звене.

Упрощения (по текущему этапу проекта):
- плоская (2D) геометрия в плоскости XZ;
- точки крепления заданы в метрах;
- цилиндр прикладывает силу вдоль оси AB (A — база, B — точка на звене).

Практика для симуляции:
- У решения по длине цилиндра часто есть две ветки ("локоть вверх/вниз").
- Если ветку выбирать только по |θ| или близости к θprev в ГСК,
  возможны скачки при движении родительского звена (повороты всей кинематической цепочки).
- Поэтому для устойчивости предусмотрен API `solve_angle_with_branch(..., branch_prev=...)`,
  где `branch_prev` — инвариантное обозначение ветки (+1/-1), которое можно хранить в состоянии.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np


BranchSign = Literal[-1, 1]


@dataclass(frozen=True)
class CylinderAttachment:
    """Параметры крепления цилиндра для решения кинематики.

    Атрибуты:
        pivot_point: (x, z) шарнир звена в ГСК (м).
        base_point: (x, z) крепление цилиндра на базовом теле в ГСК (м).
        link_point_local: (x, z) крепление штока на звене в локальной системе звена (м).
        link_length_m: Длина звена (м). На данном этапе используется только
                       для валидации (должна быть > 0).
    """

    pivot_point: Tuple[float, float]
    base_point: Tuple[float, float]
    link_point_local: Tuple[float, float]
    link_length_m: float

    def __post_init__(self) -> None:
        if self.link_length_m <= 0:
            raise ValueError("link_length_m must be > 0")


class CylinderLinkMechanism:
    """Решает кинематику и динамику механизма "цилиндр-звено"."""

    def __init__(self, att: CylinderAttachment) -> None:
        self._att = att
        self._pivot = np.array(att.pivot_point, dtype=np.float64)
        self._base = np.array(att.base_point, dtype=np.float64)
        self._link_local = np.array(att.link_point_local, dtype=np.float64)

    @property
    def attachment(self) -> CylinderAttachment:
        return self._att

    @staticmethod
    def _wrap_pi(angle: float) -> float:
        """Wrap angle to [-pi, pi] using +pi as canonical boundary value."""

        out = float((angle + np.pi) % (2.0 * np.pi) - np.pi)
        # Canonicalize boundary: represent both -pi and +pi as +pi.
        if np.isclose(out, -np.pi):
            return float(np.pi)
        return out

    def solve_angle(self, cyl_length_m: float) -> Tuple[float, float]:
        """Совместимый API: вернуть (theta_rad, dtheta_dl).

        Ветку выбирает канонически (минимальный |θ| после нормализации),
        без использования предыдущего состояния.
        """

        theta, dtheta_dl, _branch = self.solve_angle_with_branch(cyl_length_m)
        return float(theta), float(dtheta_dl)

    def solve_angle_with_branch(
        self,
        cyl_length_m: float,
        *,
        branch_prev: BranchSign | None = None,
        theta_prev_rad: float | None = None,
    ) -> Tuple[float, float, BranchSign]:
        """Решить угол и вернуть выбранную ветку.

        Args:
            cyl_length_m: длина цилиндра AB (м).
            branch_prev: ветка на предыдущем шаге (+1/-1). Это инвариантно к
                глобальному повороту механизма (движению родительского звена),
                поэтому предпочтительнее для непрерывной симуляции.
            theta_prev_rad: предыдущий θ в ГСК (опционально). Используется только
                если branch_prev не задан.

        Returns:
            (theta_rad, dtheta_dl, branch_sign)

        Примечание:
            В сингулярных случаях (sin(∠AOB) ≈ 0) возвращается dθ/dl = 0.0.
        """

        AB = float(cyl_length_m)

        OA_vec = self._base - self._pivot
        OA = float(np.linalg.norm(OA_vec))
        if OA <= 0:
            raise ValueError("Invalid geometry: OA length is 0")

        OB = float(np.linalg.norm(self._link_local))
        if OB <= 0:
            raise ValueError("Invalid geometry: OB length is 0")

        angle_OA = float(np.arctan2(OA_vec[1], OA_vec[0]))
        angle_OB_local = float(np.arctan2(self._link_local[1], self._link_local[0]))

        # Закон косинусов
        cos_aob = (OA * OA + OB * OB - AB * AB) / (2.0 * OA * OB)
        cos_aob = float(np.clip(cos_aob, -1.0, 1.0))
        aob = float(np.arccos(cos_aob))

        # Два решения: angle_OB_global = angle_OA ± ∠AOB
        theta_plus = angle_OA + aob - angle_OB_local
        theta_minus = angle_OA - aob - angle_OB_local

        theta_plus_n = self._wrap_pi(theta_plus)
        theta_minus_n = self._wrap_pi(theta_minus)

        def _dist_to_theta_prev(theta: float) -> float:
            if theta_prev_rad is None:
                return abs(theta)
            return abs(self._wrap_pi(theta - float(theta_prev_rad)))

        # Выбор ветки: 1) по branch_prev (инвариантно), 2) по theta_prev, 3) канонически
        if branch_prev in (-1, 1):
            if branch_prev == 1:
                theta = theta_plus_n
                branch: BranchSign = 1
            else:
                theta = theta_minus_n
                branch = -1
        elif theta_prev_rad is not None:
            d_plus = _dist_to_theta_prev(theta_plus_n)
            d_minus = _dist_to_theta_prev(theta_minus_n)
            if d_minus <= d_plus:
                theta = theta_minus_n
                branch = -1
            else:
                theta = theta_plus_n
                branch = 1
        else:
            if abs(theta_minus_n) <= abs(theta_plus_n):
                theta = theta_minus_n
                branch = -1
            else:
                theta = theta_plus_n
                branch = 1

        # Производная по длине цилиндра зависит от выбранной ветки
        sin_aob = float(np.sqrt(max(0.0, 1.0 - cos_aob * cos_aob)))
        if sin_aob < 1e-6:
            dtheta_dl = 0.0
        else:
            dtheta_base = AB / (OA * OB * sin_aob)
            dtheta_dl = float(branch) * dtheta_base

        return float(theta), float(dtheta_dl), branch

    def cylinder_force_to_moment(self, cyl_length_m: float, force_N: float) -> float:
        """Преобразовать силу в цилиндре в момент относительно шарнира O.

        Момент вычисляется как 2D-скалярное произведение (r × F):
            M = (OB_vec_x * F_z - OB_vec_z * F_x)
        где F направлена вдоль оси AB.
        """

        F = float(force_N)
        if abs(F) < 1e-12:
            return 0.0

        theta, _dtheta_dl = self.solve_angle(cyl_length_m)

        c = float(np.cos(theta))
        s = float(np.sin(theta))
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)

        B = self._pivot + rot @ self._link_local

        AB_vec = B - self._base
        AB_norm = float(np.linalg.norm(AB_vec))
        if AB_norm < 1e-12:
            return 0.0

        AB_dir = AB_vec / AB_norm
        OB_vec = B - self._pivot

        r_perp = float(OB_vec[0] * AB_dir[1] - OB_vec[1] * AB_dir[0])
        return F * r_perp

    def __repr__(self) -> str:
        return (
            "CylinderLinkMechanism("  # noqa: ISC003
            f"pivot={tuple(self._att.pivot_point)}, "
            f"base={tuple(self._att.base_point)}, "
            f"link_local={tuple(self._att.link_point_local)}"
            ")"
        )
