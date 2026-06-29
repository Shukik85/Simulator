from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from hydrosim_v2.core._validators import check_finite, check_positive, check_vec2
from hydrosim_v2.core.types import Vec2

BodyName = Literal["base", "boom", "arm", "bucket"]


@dataclass(frozen=True)
class Attachment2D:
    """Pin attachment point in the local CS of a body."""
    body: BodyName
    point_local: Vec2

    def __post_init__(self) -> None:
        check_vec2(f"{self.body}.point_local", self.point_local)


@dataclass(frozen=True)
class LinkGeometry:
    """Rigid link: geometry and inertia."""
    name: str
    length_m: float
    mass_kg: float
    com_local: Vec2 = (0.0, 0.0)

    def __post_init__(self) -> None:
        check_positive(f"{self.name}.length_m", self.length_m)
        check_positive(f"{self.name}.mass_kg", self.mass_kg)
        check_vec2(f"{self.name}.com_local", self.com_local)

    @property
    def inertia_kgm2(self) -> float:
        m = self.mass_kg
        L = self.length_m
        r2 = self.com_local[0] ** 2 + self.com_local[1] ** 2
        return m * L * L / 12.0 + m * r2


@dataclass(frozen=True)
class CylinderGeometry:
    """Hydraulic cylinder: static geometry only."""
    name: str
    stroke_m: float
    length_min_m: float
    bore_diameter_m: float
    rod_diameter_m: float
    base_mount: Attachment2D
    rod_mount: Attachment2D

    def __post_init__(self) -> None:
        check_positive(f"{self.name}.stroke_m", self.stroke_m)
        check_positive(f"{self.name}.length_min_m", self.length_min_m)
        check_positive(f"{self.name}.bore_diameter_m", self.bore_diameter_m)
        check_positive(f"{self.name}.rod_diameter_m", self.rod_diameter_m)
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
    """4-bar bucket linkage geometry (A-C-P-E-D).

    Cylinder (A→P), Lever (C→P), Rod (P→E), Bucket pivot D.
    E is the rod attachment point on the bucket (in bucket CS).
    """
    lever_length_m: float       # C→P
    rod_length_m: float         # P→E
    anchor_A: Vec2              # cylinder base on arm (arm CS)
    pivot_C: Vec2               # lever pivot on arm (arm CS)
    pivot_D: Vec2               # bucket pivot on arm (arm CS)
    E_local: Vec2               # rod attachment on bucket (bucket CS)
    bucket_tip_local: Vec2 = (1.35, 0.0)  # cutting edge in bucket CS

    def __post_init__(self) -> None:
        check_positive("lever_length_m", self.lever_length_m)
        check_positive("rod_length_m", self.rod_length_m)
        check_vec2("anchor_A", self.anchor_A)
        check_vec2("pivot_C", self.pivot_C)
        check_vec2("pivot_D", self.pivot_D)
        check_vec2("E_local", self.E_local)
        check_vec2("bucket_tip_local", self.bucket_tip_local)


@dataclass(frozen=True)
class BodyGeometry:
    """Complete geometry of one link + its cylinder."""
    link: LinkGeometry
    cylinder: CylinderGeometry
