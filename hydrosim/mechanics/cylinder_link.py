"""Кинематическая и динамическая связь гидроцилиндра и звена.

Решает:
- по длине цилиндра -> угол звена и его производная dθ/dl;
- по силе в цилиндре -> момент на звене.

Упрощения (по текущему этапу проекта):
- плоская (2D) геометрия в плоскости XZ;
- точки крепления заданы в метрах;
- цилиндр прикладывает силу вдоль оси AB (A — база, B — точка на звене).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


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
    """Решает кинематику и динамику механизма "цилиндр-звено".

    Физическая модель:
    - Три точки: O (шарнир), A (крепление цилиндра на базе), B (крепление штока на звене)
    - O фиксирована, B движется вместе со звеном при повороте на угол θ.
    - Длина AB = l_cyl (длина цилиндра).
    - По l_cyl нужно найти θ.

    Геометрия:
    - O: (x_O, z_O) — шарнир звена в ГСК
    - A: (x_A, z_A) — крепление цилиндра в ГСК (фиксировано)
    - B: B_local повернута на θ вокруг O
      B_global = O + Rot(θ) * B_local
    - AB: вектор от A к B, |AB| = l_cyl

    Решение треугольника OAB:
    - Известны: OA, OB (геометрия), AB (длина цилиндра)
    - Найти: угол ∠AOB и, следовательно, θ
    - Используем закон косинусов:
      AB² = OA² + OB² - 2·OA·OB·cos(∠AOB)

    Важно:
    - У треугольника обычно есть 2 решения ("локоть вверх/вниз").
    - Для стабильности (и чтобы тесты были однозначны) выбираем решение с
      минимальным |θ| после нормализации в диапазон [-π, π].
    """

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
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def solve_angle(self, cyl_length_m: float) -> Tuple[float, float]:
        """По текущей длине цилиндра вычислить угол звена и производную dθ/dl.

        Возвращает:
            (theta_rad, dtheta_dl)

        Примечание:
            dθ/dl вычисляется аналитически дифференцированием закона косинусов.
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

        if abs(theta_minus_n) <= abs(theta_plus_n):
            theta = theta_minus_n
            sign = -1.0
        else:
            theta = theta_plus_n
            sign = 1.0

        # dθ/dl = ± d(∠AOB)/d(AB)
        # cos = (OA^2 + OB^2 - AB^2)/(2 OA OB)
        # dcos/dAB = -AB/(OA OB)
        # d(arccos(cos))/dAB = AB/(OA OB sin(∠AOB))
        sin_aob = float(np.sqrt(max(0.0, 1.0 - cos_aob * cos_aob)))
        if sin_aob < 1e-6:
            dtheta_dl = 0.0
        else:
            dtheta_dl = sign * (AB / (OA * OB * sin_aob))

        return float(theta), float(dtheta_dl)

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
