from __future__ import annotations

from dataclasses import dataclass, field

from hydrosim_v2.config.base import (
    LinkGeometry, CylinderGeometry, Attachment2D,
    BucketLeverMechanismParams,
)


@dataclass(frozen=True)
class ExcavatorMechanicsConfig:
    boom_link: LinkGeometry
    arm_link: LinkGeometry
    bucket_link: LinkGeometry
    boom_cyl: CylinderGeometry
    arm_cyl: CylinderGeometry
    bucket_cyl: CylinderGeometry
    bucket_lever: BucketLeverMechanismParams

    def __post_init__(self) -> None:
        names = {self.boom_link.name, self.arm_link.name, self.bucket_link.name}
        for cyl in (self.boom_cyl, self.arm_cyl, self.bucket_cyl):
            if cyl.rod_mount.body not in names:
                raise ValueError(f"{cyl.name}: unknown rod_mount.body={cyl.rod_mount.body}")

    def links(self) -> dict[str, LinkGeometry]:
        return {
            self.boom_link.name: self.boom_link,
            self.arm_link.name: self.arm_link,
            self.bucket_link.name: self.bucket_link,
        }

    def cylinders(self) -> dict[str, CylinderGeometry]:
        return {
            self.boom_cyl.name: self.boom_cyl,
            self.arm_cyl.name: self.arm_cyl,
            self.bucket_cyl.name: self.bucket_cyl,
        }


DEFAULT_MECHANICS_CONFIG = ExcavatorMechanicsConfig(
    boom_link=LinkGeometry(
        name="boom", length_m=4.7, mass_kg=835.0,
        com_local=(2.21, 0.535),
    ),
    boom_cyl=CylinderGeometry(
        name="boom_cyl", stroke_m=1.0, length_min_m=1.5,
        bore_diameter_m=0.110, rod_diameter_m=0.080,
        base_mount=Attachment2D(body="base", point_local=(0.41, -0.415)),
        rod_mount=Attachment2D(body="boom", point_local=(1.81, 0.88)),
    ),
    arm_link=LinkGeometry(
        name="arm", length_m=2.0, mass_kg=262.0,
        com_local=(0.54, 0.165),
    ),
    arm_cyl=CylinderGeometry(
        name="arm_cyl", stroke_m=1.185, length_min_m=1.685,
        bore_diameter_m=0.110, rod_diameter_m=0.080,
        base_mount=Attachment2D(body="boom", point_local=(2.645, 0.945)),
        rod_mount=Attachment2D(body="arm", point_local=(-0.55, 0.375)),
    ),
    bucket_link=LinkGeometry(
        name="bucket", length_m=0.376, mass_kg=1200.0,
        com_local=(0.44, 0.45),
    ),
    bucket_cyl=CylinderGeometry(
        name="bucket_cyl", stroke_m=0.9, length_min_m=1.4,
        bore_diameter_m=0.100, rod_diameter_m=0.063,
        base_mount=Attachment2D(body="arm", point_local=(-0.135, 0.51)),
        rod_mount=Attachment2D(body="bucket", point_local=(-0.056503, 0.371227)),
    ),
    bucket_lever=BucketLeverMechanismParams(
        lever_length_m=0.52,
        rod_length_m=0.480,
        anchor_A=(-0.135, 0.51),
        pivot_C=(1.685, 0.0),
        pivot_D=(2.0, 0.0),
        E_local=(-0.056503, 0.371227),
        bucket_tip_local=(1.35, 0.0),
    ),
)
