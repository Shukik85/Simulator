"""mechanics.py

Data‑only configuration of the excavator mechanics (planar 2‑D model).

The module contains pure data classes describing:
* link geometry (length, mass, centre‑of‑mass);
* hydraulic cylinder geometry (stroke, min length, diameters, mounting points);
* swing mechanism parameters;
* bucket‑lever mechanism parameters.

No dynamics (friction, leakage, flow) is modelled here – the intention is to
provide a single source of truth for the mechanical layout that kinematic and
dynamic modules can consume.

Coordinate convention (2‑D):
    * X – horizontal forward direction,
    * Y – vertical upward direction,
    * rotation occurs about the Z‑axis (out‑of‑plane).
    * All attachment points are given in the **local** coordinate system of the
      body they belong to.  The kinematic layer is responsible for transforming
      these points to the world frame using the current joint angles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, Literal, Optional

# ----------------------------------------------------------------------
# Helper types & validation
# ----------------------------------------------------------------------
Vec2 = Tuple[float, float]
Body = Literal["base", "boom", "arm", "bucket"]


def _is_finite(x: float) -> bool:
    return math.isfinite(float(x))


def _check_vec2(name: str, v: Vec2) -> None:
    """Validate that a 2‑D vector contains finite numbers."""
    x, y = float(v[0]), float(v[1])
    if not (_is_finite(x) and _is_finite(y)):
        raise ValueError(f"{name} must contain finite numbers; got {v}")


# ----------------------------------------------------------------------
# Link geometry
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LinkGeometry:
    """Geometrical and inertial properties of a rigid link (boom/arm/bucket)."""

    name: str
    length_m: float
    mass_kg: float
    com_local: Vec2 = (0.0, 0.0)          # (x, y) metres from the joint (local CS)

    def __post_init__(self) -> None:
        if self.length_m <= 0.0:
            raise ValueError("length_m must be > 0")
        if self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be > 0")
        _check_vec2(f"{self.name}.com_local", self.com_local)

    @property
    def com_offset_from_pivot_m(self) -> float:
        """Euclidean distance from the joint to the centre of mass."""
        x, y = self.com_local
        return math.hypot(x, y)

    @property
    def inertia_about_pivot_kg_m2(self) -> float:
        """
        Moment of inertia about the joint.

        For a slender rod the centroidal moment (about an axis perpendicular to
        the rod and passing through its centre) is m·L²/12, independent of the
        rod’s orientation in the plane.  The parallel‑axis theorem then gives:

            J = J_cm + m·r²

        where *r* is the distance from the joint to the centre of mass.
        """
        m = float(self.mass_kg)
        r_squared = self.com_local[0] ** 2 + self.com_local[1] ** 2
        j_cm_approx = m * (self.length_m ** 2) / 12.0
        return j_cm_approx + m * r_squared

    def __repr__(self) -> str:
        x, y = self.com_local
        return (
            f"LinkGeometry(name={self.name}, L={self.length_m}m, "
            f"m={self.mass_kg}kg, com=({x:.3f}, {y:.3f})m, "
            f"J_pivot={self.inertia_about_pivot_kg_m2:.1f} kg·m²)"
        )


# ----------------------------------------------------------------------
# Attachment points
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Attachment2D:
    """Pin attachment point expressed in the local CS of a body."""

    body: Body
    point_local: Vec2

    def __post_init__(self) -> None:
        if not self.body:
            raise ValueError("body must be non‑empty")
        _check_vec2(f"{self.body}.point_local", self.point_local)


# ----------------------------------------------------------------------
# Cylinder geometry
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class CylinderGeometry:
    """Static data of a hydraulic cylinder (pin‑to‑pin)."""

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

        # Validate that the bodies referenced actually exist (checked later
        # in MechanicsConfig, but we can do a lightweight check here as well)
        if self.base_mount.body not in {"base", "boom", "arm", "bucket"}:
            raise ValueError(f"unknown base_mount.body={self.base_mount.body}")
        if self.rod_mount.body not in {"base", "boom", "arm", "bucket"}:
            raise ValueError(f"unknown rod_mount.body={self.rod_mount.body}")

    @property
    def length_max_m(self) -> float:
        """Maximum pin‑to‑pin length (fully extended)."""
        return self.length_min_m + self.stroke_m

    def length_from_extension(self, x_m: float) -> float:
        """Pin‑to‑pin length as a function of rod extension *x* (clamped)."""
        x = max(0.0, min(self.stroke_m, float(x_m)))
        return self.length_min_m + x

    def extension_from_length(self, length_m: float) -> float:
        """Rod extension *x* as a function of pin‑to‑pin length *L* (clamped)."""
        L = float(length_m)
        if L <= self.length_min_m:
            return 0.0
        if L >= self.length_max_m:
            return self.stroke_m
        return L - self.length_min_m

    @property
    def area_head_m2(self) -> float:
        """Piston‑side area (m²)."""
        d = self.bore_diameter_m
        return 0.25 * math.pi * d * d

    @property
    def area_rod_m2(self) -> float:
        """Rod cross‑sectional area (m²)."""
        d = self.rod_diameter_m
        return 0.25 * math.pi * d * d

    @property
    def area_annulus_m2(self) -> float:
        """Rod‑side area (m²)."""
        return max(1e-9, self.area_head_m2 - self.area_rod_m2)

    def __repr__(self) -> str:
        return (
            f"CylinderGeometry(name={self.name}, Lmin={self.length_min_m}m, "
            f"Lmax={self.length_max_m}m, stroke={self.stroke_m}m)"
        )


# ----------------------------------------------------------------------
# Bucket‑lever mechanism
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BucketLeverMechanismParams:
    """Parameters of the bucket‑lever (linkage) mechanism."""

    lever_length_m: float
    rod_length_m: float
    anchor_A: Vec2          # Cylinder rod end (in world/excavator base CS)
    pivot_C: Vec2           # Lever pivot (on the arm)
    pivot_D: Vec2           # Bucket pivot (on the bucket)

    def __post_init__(self) -> None:
        if self.lever_length_m <= 0:
            raise ValueError("lever_length_m must be positive")
        if self.rod_length_m <= 0:
            raise ValueError("rod_length_m must be positive")
        _check_vec2("anchor_A", self.anchor_A)
        _check_vec2("pivot_C", self.pivot_C)
        _check_vec2("pivot_D", self.pivot_D)


# ----------------------------------------------------------------------
# Swing mechanism
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SwingMechanism:
    """Simplified swing‑drive parameters."""

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
    def disp_m3_per_rad(self) -> float:
        """
        Motor displacement expressed as volume per radian.

        cc/rev → m³/rev → m³/rad (divide by 2π).
        """
        return (self.motor_displacement_cc_rev * 1e-6) / (2.0 * math.pi)


# ----------------------------------------------------------------------
# Top‑level configuration
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MechanicsConfig:
    """Complete, data‑only mechanical configuration."""

    boom_link: LinkGeometry
    arm_link: LinkGeometry
    bucket_link: LinkGeometry

    boom_cyl: CylinderGeometry
    arm_cyl: CylinderGeometry
    bucket_cyl: CylinderGeometry

    swing: SwingMechanism = field(default_factory=SwingMechanism)
    bucket_lever: Optional[BucketLeverMechanismParams] = None

    def __post_init__(self) -> None:
        # Verify that all cylinder mounts refer to known bodies.
        link_names = {self.boom_link.name, self.arm_link.name, self.bucket_link.name}
        for cyl in (self.boom_cyl, self.arm_cyl, self.bucket_cyl):
            if cyl.base_mount.body != "base" and cyl.base_mount.body not in link_names:
                raise ValueError(
                    f"{cyl.name}: unknown base_mount.body={cyl.base_mount.body}"
                )
            if cyl.rod_mount.body not in link_names:
                raise ValueError(
                    f"{cyl.name}: unknown rod_mount.body={cyl.rod_mount.body}"
                )

    # ------------------------------------------------------------------
    # Convenience views
    # ------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Default configuration (based on typical excavator data)
# ----------------------------------------------------------------------
DEFAULT_MECHANICS_CONFIG = MechanicsConfig(
    boom_link=LinkGeometry(
        name="boom",
        length_m=5.0,
        mass_kg=2000.0,
        com_local=(1.707, 0.465),
    ),
    arm_link=LinkGeometry(
        name="arm",
        length_m=3.0,
        mass_kg=1200.0,
        com_local=(0.555, 0.86),
    ),
    bucket_link=LinkGeometry(
        name="bucket",
        length_m=1.0,
        mass_kg=600.0,
        com_local=(0.6, 0.5),
    ),
    boom_cyl=CylinderGeometry(
        name="boom_cyl",
        stroke_m=3.4 - 2.0,
        length_min_m=2.0,
        bore_diameter_m=0.18,
        rod_diameter_m=0.11,
        base_mount=Attachment2D(body="base", point_local=(0.860, 1.36)),
        rod_mount=Attachment2D(body="boom", point_local=(2.765, 0.952)),
    ),
    arm_cyl=CylinderGeometry(
        name="arm_cyl",
        stroke_m=2.6 - 1.6,
        length_min_m=1.6,
        bore_diameter_m=0.11,
        rod_diameter_m=0.063,
        base_mount=Attachment2D(body="boom", point_local=(0.01, 0.42)),
        rod_mount=Attachment2D(body="arm", point_local=(1.55, 0.0)),
    ),
    bucket_cyl=CylinderGeometry(
        name="bucket_cyl",
        stroke_m=1.81 - 1.21,
        length_min_m=1.21,
        bore_diameter_m=0.11,
        rod_diameter_m=0.063,
        base_mount=Attachment2D(body="arm", point_local=(-0.105, 0.286)),
        rod_mount=Attachment2D(body="bucket", point_local=(1.5, 0.0)),
    ),
    bucket_lever=BucketLeverMechanismParams(
        lever_length_m=0.425,
        rod_length_m=0.435,
        anchor_A=(-3.160, 0.785),
        pivot_C=(-0.475, 0.0),
        pivot_D=(0.0, 0.0),
    ),
)
