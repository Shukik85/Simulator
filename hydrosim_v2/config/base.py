"""Base geometry types for the excavator model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Body = Literal["base", "boom", "arm", "bucket"]


def _check_finite(name: str, v: tuple) -> None:
    for val in v:
        if val is None or not isinstance(val, (int, float)):
            raise ValueError(f"{name} must contain finite numbers; got {v}")
    x, y = float(v[0]), float(v[1])
    if not (x == x and y == y):
        raise ValueError(f"{name} must be finite; got {v}")


def _check_positive(name: str, v: float) -> None:
    if v <= 0:
        raise ValueError(f"{name} must be positive; got {v}")


@dataclass(frozen=True)
class Attachment2D:
    body: Body
    point_local: tuple[float, float]

    def __post_init__(self) -> None:
        _check_finite(f"{self.body}.point_local", self.point_local)


@dataclass(frozen=True)
class LinkGeometry:
    name: str
    length_m: float
    mass_kg: float
    com_local: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        _check_positive(f"{self.name}.length_m", self.length_m)
        _check_positive(f"{self.name}.mass_kg", self.mass_kg)
        _check_finite(f"{self.name}.com_local", self.com_local)

    @property
    def inertia_kgm2(self) -> float:
        m = self.mass_kg
        L = self.length_m
        r2 = self.com_local[0] ** 2 + self.com_local[1] ** 2
        return m * L * L / 12.0 + m * r2


@dataclass(frozen=True)
class CylinderGeometry:
    name: str
    stroke_m: float
    length_min_m: float
    bore_diameter_m: float
    rod_diameter_m: float
    base_mount: Attachment2D
    rod_mount: Attachment2D

    def __post_init__(self) -> None:
        _check_positive(f"{self.name}.stroke_m", self.stroke_m)
        _check_positive(f"{self.name}.length_min_m", self.length_min_m)
        _check_positive(f"{self.name}.bore_diameter_m", self.bore_diameter_m)
        _check_positive(f"{self.name}.rod_diameter_m", self.rod_diameter_m)
        if self.rod_diameter_m >= self.bore_diameter_m:
            raise ValueError(f"{self.name}: rod_diameter must be < bore_diameter")

    @property
    def length_max_m(self) -> float:
        return self.length_min_m + self.stroke_m

    @property
    def area_piston_m2(self) -> float:
        d = self.bore_diameter_m
        return 0.25 * 3.141592653589793 * d * d

    @property
    def area_rod_m2(self) -> float:
        d = self.rod_diameter_m
        return 0.25 * 3.141592653589793 * d * d

    @property
    def area_annulus_m2(self) -> float:
        return max(1e-12, self.area_piston_m2 - self.area_rod_m2)


@dataclass(frozen=True)
class BucketLeverMechanismParams:
    lever_length_m: float
    rod_length_m: float
    anchor_A: tuple[float, float]
    pivot_C: tuple[float, float]
    pivot_D: tuple[float, float]
    E_local: tuple[float, float]
    bucket_tip_local: tuple[float, float] = (1.35, 0.0)

    def __post_init__(self) -> None:
        _check_positive("lever_length_m", self.lever_length_m)
        _check_positive("rod_length_m", self.rod_length_m)
        _check_finite("anchor_A", self.anchor_A)
        _check_finite("pivot_C", self.pivot_C)
        _check_finite("pivot_D", self.pivot_D)
        _check_finite("E_local", self.E_local)
        _check_finite("bucket_tip_local", self.bucket_tip_local)
