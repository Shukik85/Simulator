# tests/test_mechanics.py

import numpy as np
from hydrosim.mechanics.rotary_actuator import RotaryActuatorKinematics, HydraulicLinkageGeometry
from hydrosim.mechanics.bucket_lever import BucketLeverMechanism
from hydrosim.mechanics.kinematics import ExcavatorKinematics, ExcavatorKinematicsStepper
from hydrosim.mechanics.dynamics import ExcavatorDynamics, LinkGeometry


def test_rotary_actuator():
    """Тест простого поворотного привода (для стрелы/рукояти)."""
    geom = HydraulicLinkageGeometry(
        joint_pivot_local=(0.0, 0.0),
        force_application_point_local=(1.0, 0.0),
        tip_local=(2.0, 0.0),
        parent_link_name="boom",
        cylinder_anchoring_point_local=(0.5, 1.0),
    )
    mech = RotaryActuatorKinematics(geom)

    # Простой расчёт угла
    angle, dangle_dl = mech.solve_joint_angle(cylinder_length_m=1.2)
    assert isinstance(angle, float)
    assert isinstance(dangle_dl, float)
    assert -np.pi <= angle <= np.pi

    # Проверка ветвей
    state_plus = mech.solve_kinematics_with_branch(1.2, branch_prev=1)
    state_minus = mech.solve_kinematics_with_branch(1.2, branch_prev=-1)
    assert state_plus.branch == 1
    assert state_minus.branch == -1
    assert not np.isclose(state_plus.joint_angle_rad, state_minus.joint_angle_rad, atol=1e-6)


def test_bucket_lever():
    fixed = {"O": (0.0, 0.0), "V": (1.0, 0.0), "G": (0.0, 0.5)}
    mech = BucketLeverMechanism(
        fixed_points_m=fixed,
        initial_A_m=(0.0, 0.3),
        initial_B_m=(0.9, 0.1),
        min_cylinder_length_m=0.2,
        max_cylinder_length_m=0.8,
        default_branch=1,
    )
    theta, dtheta_dl = mech.solve_joint_angle(0.5)
    assert -np.pi <= theta <= np.pi


def test_full_kinematics():
    """Полная кинематика экскаватора с 4-звенным ковшом."""
    # Стрела (boom)
    boom_geom = HydraulicLinkageGeometry(
        joint_pivot_local=(0.0, 0.0),
        force_application_point_local=(1.0, 0.0),
        tip_local=(6.0, 0.0),
        parent_link_name="chassis",
        cylinder_anchoring_point_local=(2.0, 2.0),
    )
    boom_mech = RotaryActuatorKinematics(boom_geom)

    # Рукоять (arm)
    arm_geom = HydraulicLinkageGeometry(
        joint_pivot_local=(0.0, 0.0),
        force_application_point_local=(1.0, 0.0),
        tip_local=(4.0, 0.0),
        parent_link_name="boom",
        cylinder_anchoring_point_local=(2.0, 1.0),
    )
    arm_mech = RotaryActuatorKinematics(arm_geom)

    # Ковш (bucket) — 4-звенный механизм
    bucket_fixed = {
        "O": (0.1, -0.2),
        "V": (1.0, 0.0),
        "G": (0.8, 0.1),
    }
    bucket_mech = BucketLeverMechanism(
        fixed_points_m=bucket_fixed,
        initial_A_m=(0.15, -0.1),
        initial_B_m=(1.1, -0.05),
        min_cylinder_length_m=0.3,
        max_cylinder_length_m=0.6,
        default_branch=1,
    )

    # Кинематика
    kin = ExcavatorKinematics(boom_mech, arm_mech, bucket_mech)
    stepper = ExcavatorKinematicsStepper(kin)

    # Прямая кинематика
    state = stepper.forward(
        bc=2.1,   # длина цилиндра стрелы
        ac=1.8,   # рукояти
        bkc=0.45, # ковша
        sw=0.1,   # поворот
    )

    assert hasattr(state, "bucket_tip_xyz")
    assert len(state.bucket_tip_xyz) == 3
    assert isinstance(state.boom, type(state.arm))  # все одинаковые типы


def test_dynamics():
    """Тест полной динамики с обратным проходом."""
    # Как в test_full_kinematics
    boom_geom = HydraulicLinkageGeometry(
        joint_pivot_local=(0.0, 0.0),
        force_application_point_local=(1.0, 0.0),
        tip_local=(6.0, 0.0),
        parent_link_name="chassis",
        cylinder_anchoring_point_local=(2.0, 2.0),
    )
    boom_mech = RotaryActuatorKinematics(boom_geom)

    arm_geom = HydraulicLinkageGeometry(
        joint_pivot_local=(0.0, 0.0),
        force_application_point_local=(1.0, 0.0),
        tip_local=(4.0, 0.0),
        parent_link_name="boom",
        cylinder_anchoring_point_local=(2.0, 1.0),
    )
    arm_mech = RotaryActuatorKinematics(arm_geom)

    bucket_fixed = {"O": (0.1, -0.2), "V": (1.0, 0.0), "G": (0.8, 0.1)}
    bucket_mech = BucketLeverMechanism(
        fixed_points_m=bucket_fixed,
        initial_A_m=(0.15, -0.1),
        initial_B_m=(1.1, -0.05),
        min_cylinder_length_m=0.3,
        max_cylinder_length_m=0.6,
        default_branch=1,
    )

    kin = ExcavatorKinematics(boom_mech, arm_mech, bucket_mech)
    stepper = ExcavatorKinematicsStepper(kin)

    dyn = ExcavatorDynamics(
        kinematics_stepper=stepper,
        link_geometries={
            "boom": LinkGeometry(com_local=(3.0, 0.2), mass_kg=2000, inertia_kgm2=5000),
            "arm": LinkGeometry(com_local=(2.0, -0.1), mass_kg=1200, inertia_kgm2=2000),
            "bucket": LinkGeometry(com_local=(0.6, 0.1), mass_kg=500, inertia_kgm2=300),
        },
    )

    # Прямая динамика
    kin_state = dyn.forward(boom_cyl_m=2.1, arm_cyl_m=1.8, bucket_cyl_m=0.45, swing_rad=0.0)

    assert kin_state.swing_angle_rad == 0.0
    assert abs(kin_state.gravity_global[2] + 9.81) < 1e-6

    # Обратная динамика: нагрузка на ковш
    forces = dyn.backward(
        kin_state,
        external_forces={"bucket": (0.0, 0.0, -5000)},  # 5 кН вниз
    )

    assert "boom" in forces
    assert "arm" in forces
    assert "bucket" in forces
    assert isinstance(forces["bucket"], float)
    assert forces["bucket"] < 0  # тянет при опускании ковша под нагрузкой

    # Якобиан
    J = dyn.compute_jacobian_spatial(kin_state)
    assert J.shape == (2, 3)
    assert not np.allclose(J, 0)


if __name__ == "__main__":
    print("🔧 Running mechanics tests...")
    test_rotary_actuator()
    test_bucket_lever()
    test_full_kinematics()
    test_dynamics()
    print("✅ All tests passed!")