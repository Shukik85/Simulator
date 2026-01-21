"""Механическая конфигурация экскаватора.

Содержит:
- описание геометрии звеньев (boom, arm, bucket);
- описание геометрии цилиндров;
- расчётные свойства (момент инерции, центр масс).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class LinkGeometry:
    """Геометрия одного звена (boom / arm / bucket).

    Атрибуты:
        name: Имя звена ("boom", "arm", "bucket").
        length_m: Длина звена по оси вращения (м).
        mass_kg: Масса звена (кг).
        com_ratio: Относительное положение центра масс вдоль длины (0..1).
                   0.0 = в шарнире (proximal), 1.0 = на конце (distal).
        cyl_attach_local: (x, z) точка крепления цилиндра в локальной системе звена (м).
                          x — вдоль оси звена, z — перпендикулярно (поперёк).
    """

    name: str
    length_m: float
    mass_kg: float
    com_ratio: float = 0.4
    cyl_attach_local: Tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if self.length_m <= 0:
            raise ValueError("length_m must be > 0")
        if self.mass_kg <= 0:
            raise ValueError("mass_kg must be > 0")
        if not (0.0 <= self.com_ratio <= 1.0):
            raise ValueError("com_ratio must be in [0, 1]")

    @property
    def com_offset_m(self) -> float:
        """Положение центра масс вдоль оси звена от шарнира (м).

        Формула:
            com_offset = com_ratio * length_m
        """

        return float(self.com_ratio * self.length_m)

    @property
    def moment_of_inertia_pivot(self) -> float:
        """Момент инерции стержня относительно проксимального шарнира (кг·м²).

        Модель: тонкий однородный стержень массой m, длиной L.

        Общая формула:
            J = m * L² * (1/12 + com_ratio²)
        """

        m = float(self.mass_kg)
        L = float(self.length_m)
        r = float(self.com_ratio)
        return m * (L * L) * ((1.0 / 12.0) + (r * r))

    def __repr__(self) -> str:
        return (
            f"LinkGeometry(name={self.name}, L={self.length_m}m, "
            f"m={self.mass_kg}kg, J_pivot={self.moment_of_inertia_pivot:.1f} kg·m²)"
        )


@dataclass(frozen=True)
class CylinderGeometry:
    """Геометрия гидроцилиндра (boom / arm / bucket).

    Атрибуты:
        name: Имя цилиндра ("boom_cyl", "arm_cyl", "bucket_cyl").
        base_point_global: (x, z) точка крепления на базовом теле
                           (платформа или предыдущее звено) в ГСК (м).
        rod_attach_local: (x, z) крепление штока на текущем звене
                          в ЛОКАЛЬНОЙ системе звена (м).
        stroke_m: Полный ход штока (выдвижение) (м).
        base_length_m: Длина цилиндра при 0% хода (полностью втянут) (м).
    """

    name: str
    base_point_global: Tuple[float, float]
    rod_attach_local: Tuple[float, float]
    stroke_m: float
    base_length_m: float

    def __post_init__(self) -> None:
        if self.stroke_m <= 0:
            raise ValueError("stroke_m must be > 0")
        if self.base_length_m <= 0:
            raise ValueError("base_length_m must be > 0")

    @property
    def length_min_m(self) -> float:
        """Минимальная длина цилиндра (полностью втянут, 0% выдвижения)."""

        return float(self.base_length_m)

    @property
    def length_max_m(self) -> float:
        """Максимальная длина цилиндра (полностью выдвинут, 100% выдвижения)."""

        return float(self.base_length_m + self.stroke_m)

    def length_from_spool_position(self, spool_position: float) -> float:
        """Вычислить длину цилиндра по позиции золотника распределителя.

        Вход:
            spool_position: float, от -1.0 (полностью втянут) до +1.0 (полностью выдвинут).

        Выход:
            float: длина цилиндра в м.
        """

        u = float(spool_position)
        # Безопасное насыщение диапазона.
        u = max(-1.0, min(1.0, u))

        # [-1, 1] -> [0, 1]
        t = (u + 1.0) / 2.0
        return self.length_min_m + t * (self.length_max_m - self.length_min_m)

    def __repr__(self) -> str:
        return (
            f"CylinderGeometry(name={self.name}, "
            f"L_min={self.length_min_m}m, L_max={self.length_max_m}m)"
        )


@dataclass(frozen=True)
class MechanicsConfig:
    """Собранная конфигурация всей механики экскаватора."""

    boom_link: LinkGeometry
    arm_link: LinkGeometry
    bucket_link: LinkGeometry

    boom_cyl: CylinderGeometry
    arm_cyl: CylinderGeometry
    bucket_cyl: CylinderGeometry

    def __post_init__(self) -> None:
        for link in (self.boom_link, self.arm_link, self.bucket_link):
            x, _z = link.cyl_attach_local
            if not (0.0 <= x <= link.length_m):
                raise ValueError(
                    f"{link.name}.cyl_attach_local x must be within [0, length_m]; "
                    f"got x={x}, length_m={link.length_m}"
                )

    def __repr__(self) -> str:
        return (
            "MechanicsConfig(\n"
            f"  boom={self.boom_link}\n"
            f"  arm={self.arm_link}\n"
            f"  bucket={self.bucket_link}\n"
            ")"
        )


DEFAULT_MECHANICS_CONFIG = MechanicsConfig(
    boom_link=LinkGeometry(
        name="boom",
        length_m=5.0,
        mass_kg=1200.0,
        com_ratio=0.4,
        cyl_attach_local=(4.0, 0.2),
    ),
    arm_link=LinkGeometry(
        name="arm",
        length_m=3.0,
        mass_kg=800.0,
        com_ratio=0.38,
        cyl_attach_local=(2.5, 0.15),
    ),
    bucket_link=LinkGeometry(
        name="bucket",
        length_m=2.5,
        mass_kg=600.0,
        com_ratio=0.35,
        cyl_attach_local=(1.8, 0.1),
    ),
    boom_cyl=CylinderGeometry(
        name="boom_cyl",
        base_point_global=(0.5, 0.0),
        rod_attach_local=(4.2, 0.0),
        stroke_m=1.2,
        base_length_m=1.8,
    ),
    arm_cyl=CylinderGeometry(
        name="arm_cyl",
        base_point_global=(4.0, 0.5),
        rod_attach_local=(2.8, 0.0),
        stroke_m=0.9,
        base_length_m=1.4,
    ),
    bucket_cyl=CylinderGeometry(
        name="bucket_cyl",
        base_point_global=(2.5, 0.3),
        rod_attach_local=(2.0, 0.0),
        stroke_m=0.6,
        base_length_m=1.0,
    ),
)
