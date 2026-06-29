import numpy as np
import pytest

from hydrosim_v2.config import DEFAULT_MECHANICS_CONFIG, ExcavatorMechanicsConfig
from hydrosim_v2.kinematics import (
    forward_kinematics, solve_link_angle, backward_static,
)


@pytest.fixture
def cfg() -> ExcavatorMechanicsConfig:
    return DEFAULT_MECHANICS_CONFIG


@pytest.fixture
def cyl_lengths() -> dict[str, float]:
    return {
        "boom_cyl": 2.00,
        "arm_cyl": 2.2775,
        "bucket_cyl": 1.85,
    }


class TestSolveLinkAngle:
    def test_zero_length(self):
        theta = solve_link_angle((0.0, 0.0), (1.0, 0.0), 1.0)
        assert theta == 0.0

    def test_collinear(self):
        theta = solve_link_angle((2.0, 0.0), (1.0, 0.0), 1.0)
        assert abs(theta) < 1e-9

    def test_config_mid_stroke(self):
        from hydrosim_v2.config import DEFAULT_MECHANICS_CONFIG as cfg
        boom_cyl = cfg.cylinders()["boom_cyl"]
        L = boom_cyl.length_min_m + boom_cyl.stroke_m / 2
        A = cfg.boom_cyl.base_mount.point_local
        P = cfg.boom_cyl.rod_mount.point_local
        theta = solve_link_angle(A, P, L)
        assert np.isfinite(theta)


class TestConfig:
    def test_default_config_has_all_links(self, cfg):
        assert len(cfg.links()) == 3
        assert len(cfg.cylinders()) == 3

    def test_bucket_lever_has_E_local(self, cfg):
        assert len(cfg.bucket_lever.E_local) == 2

    def test_cylinder_lengths_sensible(self, cfg):
        for name, cyl in cfg.cylinders().items():
            assert cyl.length_max_m > cyl.length_min_m
            assert cyl.area_piston_m2 > cyl.area_annulus_m2 > 0


class TestForwardKinematics:
    def test_returns_all_keys(self, cfg, cyl_lengths):
        pts = forward_kinematics(cfg, cyl_lengths)
        expected = {
            "base", "boom_joint", "boom_tip", "arm_joint", "arm_tip",
            "bucket_joint", "bucket_tip", "bucket_tip_cutting_edge",
            "bucket_com", "A_boom", "P_boom", "A_arm", "P_arm",
        }
        assert set(pts.keys()) == expected

    def test_base_and_boom_joint_at_origin(self, cfg, cyl_lengths):
        pts = forward_kinematics(cfg, cyl_lengths)
        assert pts["base"] == (0.0, 0.0)
        assert pts["boom_joint"] == (0.0, 0.0)

    def test_all_positions_finite(self, cfg, cyl_lengths):
        pts = forward_kinematics(cfg, cyl_lengths)
        for name, pos in pts.items():
            assert all(np.isfinite(v) for v in pos), f"{name}: {pos}"

    def test_boom_tip_x_positive(self, cfg, cyl_lengths):
        pts = forward_kinematics(cfg, cyl_lengths)
        assert pts["boom_tip"][0] > 0

    def test_arm_tip_further_than_boom_tip(self, cfg, cyl_lengths):
        pts = forward_kinematics(cfg, cyl_lengths)
        arm_dist = np.linalg.norm(np.array(pts["arm_tip"]))
        boom_dist = np.linalg.norm(np.array(pts["boom_tip"]))
        assert arm_dist > boom_dist

    def test_mid_stroke_boom_positive(self, cfg, cyl_lengths):
        pts = forward_kinematics(cfg, cyl_lengths)
        assert pts["boom_tip"][1] > 0


class TestBackwardStatic:
    def test_zero_force(self, cfg, cyl_lengths):
        forces = backward_static(cfg, cyl_lengths, (0.0, 0.0))
        assert all(abs(f) < 1e-6 for f in forces.values())

    def test_vertical_force(self, cfg, cyl_lengths):
        forces = backward_static(cfg, cyl_lengths, (0.0, -1000.0))
        assert all(np.isfinite(f) for f in forces.values())
