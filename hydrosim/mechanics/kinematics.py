"""Прямая кинематика экскаватора (2D XZ + поворот платформы вокруг Z).

По длинам цилиндров вычисляет:
- углы звеньев (boom, arm, bucket);
- позиции звеньев в ГСК (глобальная система координат);
- 3D координату кончика ковша с учётом swing.

Соглашения:
- 2D плоскость: XZ (X — горизонталь, Z — вертикаль).
- swing: поворот вокруг оси Z (вертикаль). При этом X переходит в X/Y, а Z остаётся.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

import math

from hydrosim.config.mechanics import MechanicsConfig
from hydrosim.mechanics.cylinder_link import CylinderLinkMechanism


class _BucketLever(Protocol):
    def solve_angle(self, cyl_length_m: float):  # pragma: no cover
        """Должен вернуть theta (float) или (theta, dtheta_dl)."""


@dataclass
class LinkPose2D:
    """Положение одного звена в 2D плоскости (без swing)."""

    theta_rad: float
    pivot_xy: Tuple[float, float]
    com_xy: Tuple[float, float]
    tip_xy: Tuple[float, float]

    def __repr__(self) -> str:
        return (
            f"LinkPose2D(θ={math.degrees(self.theta_rad):.1f}°, "
            f"pivot={self.pivot_xy}, com={self.com_xy}, tip={self.tip_xy})"
        )


@dataclass
class ExcavatorKinematicState:
    """Полное кинематическое состояние в 2D + swing (ГСК)."""

    swing_angle_rad: float
    boom: LinkPose2D
    arm: LinkPose2D
    bucket: LinkPose2D
    bucket_tip_xyz: Tuple[float, float, float]

    def __repr__(self) -> str:
        return (
            "ExcavatorKinematicState(\n"
            f"  swing={math.degrees(self.swing_angle_rad):.1f}°\n"
            f"  boom={self.boom}\n"
            f"  arm={self.arm}\n"
            f"  bucket={self.bucket}\n"
            f"  bucket_tip_xyz={self.bucket_tip_xyz}\n"
            ")"
        )


def _theta_from_solver_result(result) -> float:
    if isinstance(result, (tuple, list)):
        return float(result[0])
    return float(result)


def _link_pose(pivot: Tuple[float, float], theta: float, length: float, com_offset: float) -> LinkPose2D:
    x0, z0 = pivot
    c = math.cos(theta)
    s = math.sin(theta)

    com = (x0 + com_offset * c, z0 + com_offset * s)
    tip = (x0 + length * c, z0 + length * s)
    return LinkPose2D(theta_rad=float(theta), pivot_xy=(float(x0), float(z0)), com_xy=com, tip_xy=tip)


class ExcavatorKinematics:
    """Класс решает прямую кинематику экскаватора."""

    def __init__(
        self,
        cfg_mech: MechanicsConfig,
        boom_mech: CylinderLinkMechanism,
        arm_mech: CylinderLinkMechanism,
        bucket_lever: _BucketLever,
    ) -> None:
        self._mech = cfg_mech
        self._boom_mech = boom_mech
        self._arm_mech = arm_mech
        self._bucket_lever = bucket_lever

    def forward(
        self,
        boom_cyl_length_m: float,
        arm_cyl_length_m: float,
        bucket_cyl_length_m: float,
        swing_angle_rad: float = 0.0,
    ) -> ExcavatorKinematicState:
        """Прямая кинематика: из длин цилиндров вычислить позиции звеньев."""

        theta_boom, _ = self._boom_mech.solve_angle(boom_cyl_length_m)
        boom_pivot = tuple(self._boom_mech.attachment.pivot_point)
        boom_pose = _link_pose(
            pivot=boom_pivot,
            theta=theta_boom,
            length=self._mech.boom_link.length_m,
            com_offset=self._mech.boom_link.com_offset_m,
        )

        theta_arm, _ = self._arm_mech.solve_angle(arm_cyl_length_m)
        arm_pose = _link_pose(
            pivot=boom_pose.tip_xy,
            theta=theta_arm,
            length=self._mech.arm_link.length_m,
            com_offset=self._mech.arm_link.com_offset_m,
        )

        theta_bucket = _theta_from_solver_result(self._bucket_lever.solve_angle(bucket_cyl_length_m))
        bucket_pose = _link_pose(
            pivot=arm_pose.tip_xy,
            theta=theta_bucket,
            length=self._mech.bucket_link.length_m,
            com_offset=self._mech.bucket_link.com_offset_m,
        )

        # 2D (X,Z) -> 3D (X,Y,Z) через swing вокруг вертикали Z.
        tip_x_2d, tip_z = bucket_pose.tip_xy
        c = math.cos(swing_angle_rad)
        s = math.sin(swing_angle_rad)
        tip_x_3d = tip_x_2d * c
        tip_y_3d = tip_x_2d * s

        state = ExcavatorKinematicState(
            swing_angle_rad=float(swing_angle_rad),
            boom=boom_pose,
            arm=arm_pose,
            bucket=bucket_pose,
            bucket_tip_xyz=(float(tip_x_3d), float(tip_y_3d), float(tip_z)),
        )
        return state

    def bucket_tip_position_xyz(self, state: ExcavatorKinematicState) -> Tuple[float, float, float]:
        return state.bucket_tip_xyz

    def __repr__(self) -> str:
        return "ExcavatorKinematics()"


class ExcavatorKinematicsStepper:
    """Шаговый решатель кинематики с памятью веток (branch_prev).

    Назначение:
    - Устойчивость к изменению ориентации родительских звеньев.
    - Устранение случайных флипов между зеркальными решениями.

    Важно:
    - Для boom/arm используется `CylinderLinkMechanism.solve_angle_with_branch()`.
    - Для bucket используется `solve_angle_with_branch()` если он есть у bucket_lever;
      иначе используется обычный `solve_angle()` (без памяти).
    """

    def __init__(
        self,
        kin: ExcavatorKinematics,
        *,
        boom_branch_prev: int = 1,
        arm_branch_prev: int = 1,
        bucket_branch_prev: int = 1,
    ) -> None:
        self._kin = kin
        self._boom_branch_prev = 1 if boom_branch_prev >= 0 else -1
        self._arm_branch_prev = 1 if arm_branch_prev >= 0 else -1
        self._bucket_branch_prev = 1 if bucket_branch_prev >= 0 else -1

    @property
    def branches(self) -> Tuple[int, int, int]:
        return (self._boom_branch_prev, self._arm_branch_prev, self._bucket_branch_prev)

    def forward(
        self,
        boom_cyl_length_m: float,
        arm_cyl_length_m: float,
        bucket_cyl_length_m: float,
        swing_angle_rad: float = 0.0,
    ) -> ExcavatorKinematicState:
        boom_mech = self._kin._boom_mech
        arm_mech = self._kin._arm_mech
        bucket_lever = self._kin._bucket_lever

        theta_boom, _dtheta, boom_branch = boom_mech.solve_angle_with_branch(
            boom_cyl_length_m,
            branch_prev=self._boom_branch_prev,
        )
        self._boom_branch_prev = int(boom_branch)

        boom_pivot = tuple(boom_mech.attachment.pivot_point)
        boom_pose = _link_pose(
            pivot=boom_pivot,
            theta=theta_boom,
            length=self._kin._mech.boom_link.length_m,
            com_offset=self._kin._mech.boom_link.com_offset_m,
        )

        theta_arm, _dtheta, arm_branch = arm_mech.solve_angle_with_branch(
            arm_cyl_length_m,
            branch_prev=self._arm_branch_prev,
        )
        self._arm_branch_prev = int(arm_branch)

        arm_pose = _link_pose(
            pivot=boom_pose.tip_xy,
            theta=theta_arm,
            length=self._kin._mech.arm_link.length_m,
            com_offset=self._kin._mech.arm_link.com_offset_m,
        )

        if hasattr(bucket_lever, "solve_angle_with_branch"):
            theta_bucket, _dtheta, bucket_branch = bucket_lever.solve_angle_with_branch(
                bucket_cyl_length_m,
                branch_prev=self._bucket_branch_prev,
            )
            self._bucket_branch_prev = int(bucket_branch)
        else:
            theta_bucket = _theta_from_solver_result(bucket_lever.solve_angle(bucket_cyl_length_m))

        bucket_pose = _link_pose(
            pivot=arm_pose.tip_xy,
            theta=theta_bucket,
            length=self._kin._mech.bucket_link.length_m,
            com_offset=self._kin._mech.bucket_link.com_offset_m,
        )

        tip_x_2d, tip_z = bucket_pose.tip_xy
        c = math.cos(swing_angle_rad)
        s = math.sin(swing_angle_rad)
        tip_x_3d = tip_x_2d * c
        tip_y_3d = tip_x_2d * s

        return ExcavatorKinematicState(
            swing_angle_rad=float(swing_angle_rad),
            boom=boom_pose,
            arm=arm_pose,
            bucket=bucket_pose,
            bucket_tip_xyz=(float(tip_x_3d), float(tip_y_3d), float(tip_z)),
        )
