import pytest

from hydrosim_v2.config import (
    DEFAULT_MECHANICS_CONFIG, ExcavatorMechanicsConfig,
    LSConfig, ExcavatorConfig,
)


class TestExcavatorMechanicsConfig:
    def test_default_links(self):
        cfg = DEFAULT_MECHANICS_CONFIG
        assert len(cfg.links()) == 3
        assert "boom" in cfg.links()
        assert "arm" in cfg.links()
        assert "bucket" in cfg.links()

    def test_default_cylinders(self):
        cfg = DEFAULT_MECHANICS_CONFIG
        assert len(cfg.cylinders()) == 3
        for name in ("boom_cyl", "arm_cyl", "bucket_cyl"):
            assert name in cfg.cylinders()

    def test_cylinder_stroke_reasonable(self):
        cfg = DEFAULT_MECHANICS_CONFIG
        for cyl in cfg.cylinders().values():
            assert 0.5 < cyl.stroke_m < 2.0
            assert cyl.length_max_m > cyl.length_min_m

    def test_bucket_lever_params(self):
        cfg = DEFAULT_MECHANICS_CONFIG
        lever = cfg.bucket_lever
        assert lever.lever_length_m > 0
        assert lever.rod_length_m > 0


class TestLSConfig:
    def test_default_config(self):
        cfg = LSConfig()
        assert cfg.pump.max_displacement_cc_rev == 70.0
        assert "boom" in cfg.valve_sections
        assert "arm" in cfg.valve_sections
        assert "bucket" in cfg.valve_sections

    def test_valve_sections_q_nom(self):
        cfg = LSConfig()
        assert cfg.valve_sections["boom"].q_nom_lpm_at_dp == 60.0
        assert cfg.valve_sections["bucket"].q_nom_lpm_at_dp == 40.0


class TestExcavatorConfig:
    def test_full_config_creation(self):
        cfg = ExcavatorConfig(
            mechanics=DEFAULT_MECHANICS_CONFIG,
            hydraulics=LSConfig(),
        )
        assert cfg.mechanics is DEFAULT_MECHANICS_CONFIG
        assert cfg.hydraulics is not None
        assert cfg.soil.base_resistance_N > 0
