import math

import pytest

from hydrosim.config.mechanics import CylinderGeometry, LinkGeometry, MechanicsConfig
from hydrosim.mechanics.cylinder_link import CylinderAttachment, CylinderLinkMechanism
from hydrosim.mechanics.kinematics import ExcavatorKinematics


class DummyBucketLever:
    def __init__(self, theta_rad: float = 0.0) -> None:
        self._theta = float(theta_rad)

    def solve_angle(self, _cyl_length_m: float):
        # Допускаем как float, так и (theta, dtheta_dl) — kinematics поддерживает оба варианта.
        return self._theta


@pytest.fixture()
def kin() -> ExcavatorKinematics:
    cfg = MechanicsConfig(
        boom_link=LinkGeometry(name="boom", length_m=2.0, mass_kg=1.0, com_ratio=0.5),
        arm_link=LinkGeometry(name="arm", length_m=1.5, mass_kg=1.0, com_ratio=0.5),
        bucket_link=LinkGeometry(name="bucket", length_m=1.0, mass_kg=1.0, com_ratio=0.5),
        boom_cyl=CylinderGeometry(
            name="boom_cyl",
            base_point_global=(0.0, 1.0),
            rod_attach_local=(1.0, 0.0),
            stroke_m=1.0,
            base_length_m=1.0,
        ),
        arm_cyl=CylinderGeometry(
            name="arm_cyl",
            base_point_global=(0.0, 1.0),
            rod_attach_local=(1.0, 0.0),
            stroke_m=1.0,
            base_length_m=1.0,
        ),
        bucket_cyl=CylinderGeometry(
            name="bucket_cyl",
            base_point_global=(0.0, 1.0),
            rod_attach_local=(1.0, 0.0),
            stroke_m=1.0,
            base_length_m=1.0,
        ),
    )

    boom_mech = CylinderLinkMechanism(
        CylinderAttachment(
            pivot_point=(0.0, 0.0),
            base_point=(0.0, 1.0),
            link_point_local=(1.0, 0.0),
            link_length_m=cfg.boom_link.length_m,
        )
    )
    arm_mech = CylinderLinkMechanism(
        CylinderAttachment(
            pivot_point=(0.0, 0.0),
            base_point=(0.0, 1.0),
            link_point_local=(1.0, 0.0),
            link_length_m=cfg.arm_link.length_m,
        )
    )

    return ExcavatorKinematics(
        cfg_mech=cfg,
        boom_mech=boom_mech,
        arm_mech=arm_mech,
        bucket_lever=DummyBucketLever(theta_rad=0.0),
    )


class TestKinematics:
    def test_forward_returns_state(self, kin: ExcavatorKinematics) -> None:
        # Для данной простой геометрии при AB=sqrt(2) получаем θ≈0.
        state = kin.forward(
            boom_cyl_length_m=math.sqrt(2.0),
            arm_cyl_length_m=math.sqrt(2.0),
            bucket_cyl_length_m=1.0,
            swing_angle_rad=0.0,
        )

        assert state.boom.pivot_xy == pytest.approx((0.0, 0.0))
        assert abs(state.boom.theta_rad) < 0.25
        assert state.arm.pivot_xy == pytest.approx(state.boom.tip_xy)
        assert state.bucket.pivot_xy == pytest.approx(state.arm.tip_xy)

    def test_forward_swing_rotates_xy(self, kin: ExcavatorKinematics) -> None:
        state0 = kin.forward(
            boom_cyl_length_m=math.sqrt(2.0),
            arm_cyl_length_m=math.sqrt(2.0),
            bucket_cyl_length_m=1.0,
            swing_angle_rad=0.0,
        )
        state90 = kin.forward(
            boom_cyl_length_m=math.sqrt(2.0),
            arm_cyl_length_m=math.sqrt(2.0),
            bucket_cyl_length_m=1.0,
            swing_angle_rad=math.pi / 2,
        )

        x0, y0, z0 = state0.bucket_tip_xyz
        x90, y90, z90 = state90.bucket_tip_xyz

        assert z90 == pytest.approx(z0)
        # При повороте на 90° весь "радиус" уходит в Y.
        assert abs(x90) < 1e-6
        assert y90 == pytest.approx(x0, abs=1e-6)
        assert y0 == pytest.approx(0.0)

    def test_bucket_tip_position_xyz(self, kin: ExcavatorKinematics) -> None:
        state = kin.forward(
            boom_cyl_length_m=math.sqrt(2.0),
            arm_cyl_length_m=math.sqrt(2.0),
            bucket_cyl_length_m=1.0,
            swing_angle_rad=0.0,
        )
        assert kin.bucket_tip_position_xyz(state) == pytest.approx(state.bucket_tip_xyz)
