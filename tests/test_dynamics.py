import math
from dataclasses import dataclass

import numpy as np
import pytest

from hydrosim.config.mechanics import CylinderGeometry, LinkGeometry, MechanicsConfig
from hydrosim.mechanics.cylinder_link import CylinderAttachment, CylinderLinkMechanism
from hydrosim.mechanics.kinematics import ExcavatorKinematics
from hydrosim.physics.dynamics import DynamicState, ExcavatorDynamics
from hydrosim.physics.hydraulics import HydraulicConfig, HydraulicModel
from hydrosim.physics.load_model import LoadModel, SoilConfig


class DummyBucketLever:
    def __init__(self, theta_rad: float = 0.0) -> None:
        self._theta = float(theta_rad)

    def solve_angle(self, _cyl_length_m: float):
        return self._theta


@dataclass
class DummyScenario:
    soil_factor: float = 1.0


@pytest.fixture()
def dynamics() -> ExcavatorDynamics:
    cfg = MechanicsConfig(
        boom_link=LinkGeometry(name="boom", length_m=2.0, mass_kg=1000.0, com_ratio=0.5),
        arm_link=LinkGeometry(name="arm", length_m=1.5, mass_kg=800.0, com_ratio=0.5),
        bucket_link=LinkGeometry(name="bucket", length_m=1.0, mass_kg=600.0, com_ratio=0.5),
        boom_cyl=CylinderGeometry(
            name="boom_cyl",
            base_point_global=(0.0, 1.0),
            rod_attach_local=(2.0, 0.0),
            stroke_m=1.0,
            base_length_m=1.0,
        ),
        arm_cyl=CylinderGeometry(
            name="arm_cyl",
            base_point_global=(0.0, 1.0),
            rod_attach_local=(1.5, 0.0),
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
            link_point_local=(2.0, 0.0),
            link_length_m=cfg.boom_link.length_m,
        )
    )
    arm_mech = CylinderLinkMechanism(
        CylinderAttachment(
            pivot_point=(0.0, 0.0),
            base_point=(0.0, 1.0),
            link_point_local=(1.5, 0.0),
            link_length_m=cfg.arm_link.length_m,
        )
    )
    bucket_mech = CylinderLinkMechanism(
        CylinderAttachment(
            pivot_point=(0.0, 0.0),
            base_point=(0.0, 1.0),
            link_point_local=(1.0, 0.0),
            link_length_m=cfg.bucket_link.length_m,
        )
    )

    # Kinematics для LoadModel использует только solve_angle() bucket_lever.
    kin = ExcavatorKinematics(
        cfg_mech=cfg,
        boom_mech=boom_mech,
        arm_mech=arm_mech,
        bucket_lever=DummyBucketLever(theta_rad=0.0),
    )
    load = LoadModel(mech_cfg=cfg, soil_cfg=SoilConfig(randomness=0.0), kin=kin)

    hyd = HydraulicModel(
        HydraulicConfig(
            p_max_bar=350.0,
            response_rate_per_s=20.0,
            cyl_motion_gain_Pa_per_m=1.0e7,
            area_boom_m2=0.005,
            area_arm_m2=0.004,
            area_bucket_m2=0.003,
        )
    )

    return ExcavatorDynamics(
        mech_cfg=cfg,
        hyd_model=hyd,
        load_model=load,
        boom_mech=boom_mech,
        arm_mech=arm_mech,
        bucket_lever=bucket_mech,
        scenario_profile=DummyScenario(soil_factor=1.0),
    )


class TestDynamicState:
    def test_roundtrip_vector(self) -> None:
        s = DynamicState(
            theta_swing=0.1,
            theta_boom=0.2,
            theta_arm=0.3,
            theta_bucket=0.4,
            omega_swing=1.0,
            omega_boom=2.0,
            omega_arm=3.0,
            omega_bucket=4.0,
            cyl_boom_length=1.1,
            cyl_arm_length=1.2,
            cyl_bucket_length=1.3,
            pressure_boom=1e6,
            pressure_arm=2e6,
            pressure_bucket=3e6,
        )
        y = s.to_vector()
        s2 = DynamicState.from_vector(y)
        assert s2 == s


class TestDynamicsRhs:
    def test_rhs_shape_and_pressure_increase(self, dynamics: ExcavatorDynamics) -> None:
        # Геометрия: при AB=sqrt(5) для boom и AB=sqrt( (1.5)^2 + 1^2 ) для arm
        y0 = DynamicState(
            theta_swing=0.0,
            theta_boom=0.0,
            theta_arm=0.0,
            theta_bucket=0.0,
            omega_swing=0.0,
            omega_boom=0.0,
            omega_arm=0.0,
            omega_bucket=0.0,
            cyl_boom_length=math.sqrt(5.0),
            cyl_arm_length=math.sqrt(1.5**2 + 1.0**2),
            cyl_bucket_length=math.sqrt(2.0),
            pressure_boom=0.0,
            pressure_arm=0.0,
            pressure_bucket=0.0,
        ).to_vector()

        dy = dynamics.rhs(t=0.0, y=y0, inputs={"boom_spool": 1.0, "arm_spool": 0.0, "bucket_spool": 0.0})
        assert isinstance(dy, np.ndarray)
        assert dy.shape == (14,)
        # spool=1 -> целевое давление Pmax, значит dP/dt должно быть > 0
        assert dy[11] > 0.0

    def test_rhs_produces_angular_acceleration_from_pressure(self, dynamics: ExcavatorDynamics) -> None:
        # Задаём давление на boom, чтобы появился момент цилиндра.
        y0 = DynamicState(
            theta_swing=0.0,
            theta_boom=0.0,
            theta_arm=0.0,
            theta_bucket=0.0,
            omega_swing=0.0,
            omega_boom=0.0,
            omega_arm=0.0,
            omega_bucket=0.0,
            cyl_boom_length=math.sqrt(5.0),
            cyl_arm_length=math.sqrt(1.5**2 + 1.0**2),
            cyl_bucket_length=math.sqrt(2.0),
            pressure_boom=0.5 * 350.0e5,
            pressure_arm=0.0,
            pressure_bucket=0.0,
        ).to_vector()

        dy = dynamics.rhs(t=0.0, y=y0, inputs={"boom_spool": 0.0, "arm_spool": 0.0, "bucket_spool": 0.0})
        domega_boom = float(dy[5])
        assert abs(domega_boom) > 0.0

    def test_pressure_is_effectively_clamped_in_force_path(self, dynamics: ExcavatorDynamics) -> None:
        # P выше Pmax -> force вычисляется по Pmax (косвенно проверим по знаку dP/dt: он должен быть < 0,
        # т.к. target=0 при spool=0, а P clamp внутри модели = Pmax).
        y0 = DynamicState(
            theta_swing=0.0,
            theta_boom=0.0,
            theta_arm=0.0,
            theta_bucket=0.0,
            omega_swing=0.0,
            omega_boom=0.0,
            omega_arm=0.0,
            omega_bucket=0.0,
            cyl_boom_length=math.sqrt(5.0),
            cyl_arm_length=math.sqrt(1.5**2 + 1.0**2),
            cyl_bucket_length=math.sqrt(2.0),
            pressure_boom=1e9,
            pressure_arm=0.0,
            pressure_bucket=0.0,
        ).to_vector()

        dy = dynamics.rhs(t=0.0, y=y0, inputs={"boom_spool": 0.0})
        assert dy[11] < 0.0
