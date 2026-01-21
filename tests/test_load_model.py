import math
from dataclasses import dataclass

import pytest

from hydrosim.config.mechanics import CylinderGeometry, LinkGeometry, MechanicsConfig
from hydrosim.mechanics.cylinder_link import CylinderAttachment, CylinderLinkMechanism
from hydrosim.mechanics.kinematics import ExcavatorKinematics
from hydrosim.physics.load_model import LoadModel, SoilConfig


class DummyBucketLever:
    def __init__(self, theta_rad: float) -> None:
        self._theta = float(theta_rad)

    def solve_angle(self, _cyl_length_m: float):
        return self._theta


@dataclass
class DummyScenario:
    soil_factor: float = 1.0


@dataclass
class DummyState:
    cyl_boom_length: float
    cyl_arm_length: float
    cyl_bucket_length: float
    omega_bucket: float = 0.0
    theta_swing: float = 0.0


@pytest.fixture()
def model() -> LoadModel:
    cfg = MechanicsConfig(
        boom_link=LinkGeometry(name="boom", length_m=2.0, mass_kg=1000.0, com_ratio=0.5),
        arm_link=LinkGeometry(name="arm", length_m=1.5, mass_kg=800.0, com_ratio=0.5),
        bucket_link=LinkGeometry(name="bucket", length_m=1.0, mass_kg=600.0, com_ratio=0.5),
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

    kin = ExcavatorKinematics(
        cfg_mech=cfg,
        boom_mech=boom_mech,
        arm_mech=arm_mech,
        bucket_lever=DummyBucketLever(theta_rad=-math.pi / 2),
    )

    soil = SoilConfig(
        base_resistance_N=25000.0,
        vel_gain_N_per_m_s=8000.0,
        penetration_gain_N_per_m=60000.0,
        penetration_exp=1.5,
        angle_factor_max=2.0,
        optimal_attack_angle_deg=30.0,
        randomness=0.0,
    )

    return LoadModel(mech_cfg=cfg, soil_cfg=soil, kin=kin)


class TestSoilResistance:
    def test_angle_factor_clamped(self, model: LoadModel) -> None:
        # При сильном отклонении cos может дать отрицательное, но clamp должен оставить >= 0.5.
        F = model.soil_resistance(
            theta_bucket=math.radians(180.0),
            v_bucket_tip_m_s=0.0,
            penetration_depth_m=0.0,
            soil_factor=1.0,
        )
        assert F >= 25000.0 * 0.5

    def test_increases_with_depth_and_speed(self, model: LoadModel) -> None:
        F0 = model.soil_resistance(theta_bucket=math.radians(30.0), v_bucket_tip_m_s=0.0, penetration_depth_m=0.0)
        F1 = model.soil_resistance(theta_bucket=math.radians(30.0), v_bucket_tip_m_s=1.0, penetration_depth_m=0.5)
        assert F1 > F0


class TestGravityMoment:
    def test_gravity_moment_formula(self, model: LoadModel) -> None:
        m = 1000.0
        d = 2.0
        theta = math.pi / 4
        expected = m * 9.81 * d * math.sin(theta)
        assert model.gravity_moment_link(m, d, theta) == pytest.approx(expected)


class TestExternalMoments:
    def test_external_moments_bucket_has_soil_when_in_ground(self, model: LoadModel) -> None:
        state = DummyState(
            cyl_boom_length=math.sqrt(2.0),
            cyl_arm_length=math.sqrt(2.0),
            cyl_bucket_length=1.0,
            omega_bucket=1.0,
            theta_swing=0.0,
        )
        moments = model.external_moments(state=state, scenario_profile=DummyScenario(soil_factor=1.0), inputs={})
        assert set(moments.keys()) == {"boom", "arm", "bucket"}
        # boom/arm только гравитация (в этой геометрии θ≈0 => момент ≈ 0)
        assert abs(moments["boom"]) < 1e-3
        assert abs(moments["arm"]) < 1e-3
        # bucket должен получить добавку от грунта
        assert abs(moments["bucket"]) > 1.0
