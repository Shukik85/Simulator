"""Механическая конфигурация экскаватора (плоская 2D-модель).

Этот модуль намеренно является data-only ("мёртвым" конфигом):
- геометрия звеньев (длина, масса, положение центра масс);
- геометрия гидроцилиндров (ход, минимальная длина, диаметры, точки крепления);
- параметры поворотного механизма (swing) в упрощённом виде.

Ключевое правило:
- Команда золотника НЕ задаёт положение/длину цилиндра.
  Положение/длина цилиндра — это состояние, получаемое из динамики
  (поток -> скорость -> интегрирование), а не прямое отображение команды.

Соглашение о координатах (2D):
- каждая точка крепления задаётся в ЛОКАЛЬНОЙ СК тела (base/boom/arm/bucket);
- кинематический модуль обязан уметь переводить эти точки в мировую СК
  по текущим углам/позам звеньев;
- длина цилиндра в момент времени = расстояние между двумя pin-точками
  в мировой СК.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import math


Vec2 = Tuple[float, float]


def _is_finite(x: float) -> bool:
    return math.isfinite(float(x))


def _check_vec2(name: str, v: Vec2) -> None:
    x, z = float(v[0]), float(v[1])
    if not (_is_finite(x) and _is_finite(z)):
        raise ValueError(f"{name} must contain finite numbers; got {v}")


@dataclass(frozen=True)
class LinkGeometry:
    """Геометрия и сосредоточенные массовые свойства звена (boom/arm/bucket)."""

    name: str
    length_m: float
    mass_kg: float
    com_ratio: float = 0.4  # 0..1 вдоль оси звена от шарнира к концу

    def __post_init__(self) -> None:
        if self.length_m <= 0.0:
            raise ValueError("length_m must be > 0")
        if self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be > 0")
        if not (0.0 <= self.com_ratio <= 1.0):
            raise ValueError("com_ratio must be in [0, 1]")

    @property
    def com_offset_m(self) -> float:
        """Смещение центра масс вдоль оси звена от шарнира (м)."""

        return float(self.com_ratio * self.length_m)

    @property
    def inertia_about_pivot_kg_m2(self) -> float:
        """Приближённый момент инерции относительно шарнира (кг·м²).

        Аппроксимация тонким стержнем:
            J_pivot = m * (L^2/12 + r^2), где r = com_offset_m
        """

        m = float(self.mass_kg)
        L = float(self.length_m)
        r = float(self.com_offset_m)
        return float(m * ((L * L) / 12.0 + r * r))

    def __repr__(self) -> str:
        return (
            f"LinkGeometry(name={self.name}, L={self.length_m}m, "
            f"m={self.mass_kg}kg, J_pivot={self.inertia_about_pivot_kg_m2:.1f} kg·m²)"
        )


@dataclass(frozen=True)
class Attachment2D:
    """Точка крепления (pin) на некотором теле, заданная в локальной СК тела."""

    body: str
    point_local: Vec2

    def __post_init__(self) -> None:
        if not self.body:
            raise ValueError("body must be non-empty")
        _check_vec2(f"{self.body}.point_local", self.point_local)


@dataclass(frozen=True)
class CylinderGeometry:
    """Геометрия гидроцилиндра + крепления pin-to-pin.

    Параметры:
        stroke_m: ход штока (м).
        length_min_m: минимальная pin-to-pin длина (полностью втянут).
        base_mount/rod_mount: точки крепления на двух телах.
        bore/rod diameters: геометрия для расчёта площадей камер.

    Важно:
        Этот класс НЕ содержит динамики (трение, утечки, расходы).
    """

    name: str

    stroke_m: float
    length_min_m: float

    base_mount: Attachment2D
    rod_mount: Attachment2D

    bore_diameter_m: float
    rod_diameter_m: float

    def __post_init__(self) -> None:
        if self.stroke_m <= 0.0:
            raise ValueError("stroke_m must be > 0")
        if self.length_min_m <= 0.0:
            raise ValueError("length_min_m must be > 0")
        if self.bore_diameter_m <= 0.0:
            raise ValueError("bore_diameter_m must be > 0")
        if self.rod_diameter_m <= 0.0:
            raise ValueError("rod_diameter_m must be > 0")
        if self.rod_diameter_m >= self.bore_diameter_m:
            raise ValueError("rod_diameter_m must be < bore_diameter_m")

    @property
    def length_max_m(self) -> float:
        return float(self.length_min_m + self.stroke_m)

    def length_from_extension(self, x_m: float) -> float:
        """Pin-to-pin длина как функция выдвижения штока x (м), x in [0, stroke]."""

        x = float(x_m)
        if x < 0.0:
            x = 0.0
        elif x > self.stroke_m:
            x = self.stroke_m
        return float(self.length_min_m + x)

    def extension_from_length(self, length_m: float) -> float:
        """Выдвижение штока x (м) как функция pin-to-pin длины L."""

        L = float(length_m)
        if L <= self.length_min_m:
            return 0.0
        if L >= self.length_max_m:
            return float(self.stroke_m)
        return float(L - self.length_min_m)

    @property
    def area_head_m2(self) -> float:
        """Площадь поршневой полости (м²)."""

        d = float(self.bore_diameter_m)
        return float(0.25 * math.pi * d * d)

    @property
    def area_rod_m2(self) -> float:
        """Площадь штока (м²)."""

        d = float(self.rod_diameter_m)
        return float(0.25 * math.pi * d * d)

    @property
    def area_annulus_m2(self) -> float:
        """Площадь штоковой полости (м²)."""

        return float(max(1e-9, self.area_head_m2 - self.area_rod_m2))

    def __repr__(self) -> str:
        return (
            f"CylinderGeometry(name={self.name}, Lmin={self.length_min_m}m, "
            f"Lmax={self.length_max_m}m, stroke={self.stroke_m}m)"
        )


@dataclass(frozen=True)
class SwingMechanism:
    """Упрощённые параметры поворотного механизма (swing)."""

    inertia_kg_m2: float = 8000.0
    gear_ratio: float = 50.0
    motor_displacement_cc_rev: float = 35.0
    coulomb_friction_nm: float = 200.0
    viscous_damping_nm_s_rad: float = 80.0

    def __post_init__(self) -> None:
        if self.inertia_kg_m2 <= 0.0:
            raise ValueError("inertia_kg_m2 must be > 0")
        if self.gear_ratio <= 0.0:
            raise ValueError("gear_ratio must be > 0")
        if self.motor_displacement_cc_rev <= 0.0:
            raise ValueError("motor_displacement_cc_rev must be > 0")

    @property
    def disp_m3_rad(self) -> float:
        # cc/rev -> m3/rev, rev -> 2*pi rad
        return float((self.motor_displacement_cc_rev * 1e-6) / (2.0 * math.pi))


@dataclass(frozen=True)
class MechanicsConfig:
    """Полный механический конфиг (data-only)."""

    boom_link: LinkGeometry
    arm_link: LinkGeometry
    bucket_link: LinkGeometry

    boom_cyl: CylinderGeometry
    arm_cyl: CylinderGeometry
    bucket_cyl: CylinderGeometry

    swing: SwingMechanism = SwingMechanism()

    def __post_init__(self) -> None:
        # Не навязываем ограничение вида "0 <= x <= length" для креплений:
        # реальные кронштейны/рычажные механизмы (особенно ковш) могут давать
        # точки вне "оси" звена. Здесь валидируем только согласованность имён тел.
        link_names = {self.boom_link.name, self.arm_link.name, self.bucket_link.name}

        for cyl in (self.boom_cyl, self.arm_cyl, self.bucket_cyl):
            if cyl.base_mount.body != "base" and cyl.base_mount.body not in link_names:
                raise ValueError(f"{cyl.name}: unknown base_mount.body={cyl.base_mount.body}")
            if cyl.rod_mount.body not in link_names:
                raise ValueError(f"{cyl.name}: unknown rod_mount.body={cyl.rod_mount.body}")

    def links(self) -> Dict[str, LinkGeometry]:
        return {
            self.boom_link.name: self.boom_link,
            self.arm_link.name: self.arm_link,
            self.bucket_link.name: self.bucket_link,
        }

    def cylinders(self) -> Dict[str, CylinderGeometry]:
        return {
            self.boom_cyl.name: self.boom_cyl,
            self.arm_cyl.name: self.arm_cyl,
            self.bucket_cyl.name: self.bucket_cyl,
        }


# Базовая конфигурация (числа близки к текущим дефолтам в проекте).
# ВАЖНО: base_mount.body="base" подразумевает, что base-пин задан в базовой СК.

DEFAULT_MECHANICS_CONFIG = MechanicsConfig(
    boom_link=LinkGeometry(name="boom", length_m=5.0, mass_kg=1200.0, com_ratio=0.4),
    arm_link=LinkGeometry(name="arm", length_m=3.0, mass_kg=800.0, com_ratio=0.38),
    bucket_link=LinkGeometry(name="bucket", length_m=2.5, mass_kg=600.0, com_ratio=0.35),

    boom_cyl=CylinderGeometry(
        name="boom_cyl",
        stroke_m=1.2,
        length_min_m=1.8,
        bore_diameter_m=0.090,
        rod_diameter_m=0.050,
        base_mount=Attachment2D(body="base", point_local=(0.5, 0.0)),
        rod_mount=Attachment2D(body="boom", point_local=(4.2, 0.0)),
    ),
    arm_cyl=CylinderGeometry(
        name="arm_cyl",
        stroke_m=0.9,
        length_min_m=1.4,
        bore_diameter_m=0.080,
        rod_diameter_m=0.045,
        base_mount=Attachment2D(body="base", point_local=(4.0, 0.5)),
        rod_mount=Attachment2D(body="arm", point_local=(2.8, 0.0)),
    ),
    bucket_cyl=CylinderGeometry(
        name="bucket_cyl",
        stroke_m=0.6,
        length_min_m=1.0,
        bore_diameter_m=0.070,
        rod_diameter_m=0.040,
        base_mount=Attachment2D(body="base", point_local=(2.5, 0.3)),
        rod_mount=Attachment2D(body="bucket", point_local=(2.0, 0.0)),
    ),
)
