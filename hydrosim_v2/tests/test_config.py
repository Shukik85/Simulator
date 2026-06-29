from hydrosim_v2.config import (
    DEFAULT_MECHANICS_CONFIG, ExcavatorMechanicsConfig,
    LSPumpConfig, LSValveSectionConfig, LSConfig,
    ExcavatorConfig, SoilConfig, Attachment2D, LinkGeometry,
    CylinderGeometry, BucketLeverMechanismParams,
)
from hydrosim_v2.core.units import bar_to_Pa, Pa_to_bar, LPM_to_m3s


def test_default_mechanics_config_creates():
    cfg = DEFAULT_MECHANICS_CONFIG
    assert cfg.boom_link.name == "boom"
    assert cfg.arm_link.name == "arm"
    assert cfg.bucket_link.name == "bucket"
    assert len(cfg.cylinders()) == 3
    assert len(cfg.links()) == 3


def test_cylinder_areas():
    cyl = DEFAULT_MECHANICS_CONFIG.boom_cyl
    assert cyl.area_piston_m2 > cyl.area_annulus_m2 > 0
    assert cyl.length_max_m > cyl.length_min_m


def test_bucket_lever_constructor():
    bl = BucketLeverMechanismParams(
        lever_length_m=0.52,
        rod_length_m=0.48,
        anchor_A=(-0.135, 0.51),
        pivot_C=(1.685, 0.0),
        pivot_D=(2.0, 0.0),
        E_local=(-0.057, 0.371),
    )
    assert bl.lever_length_m == 0.52


def test_ls_config_defaults():
    cfg = LSConfig()
    assert "boom" in cfg.valve_sections


def test_excavator_config_assembly():
    cfg = ExcavatorConfig(
        mechanics=DEFAULT_MECHANICS_CONFIG,
        hydraulics=LSConfig(),
        soil=SoilConfig(),
    )
    assert cfg.mechanics.boom_cyl.stroke_m == 1.0


def test_attachment_validation():
    att = Attachment2D(body="boom", point_local=(1.0, 2.0))
    assert att.body == "boom"


def test_link_inertia():
    link = LinkGeometry(name="t", length_m=2.0, mass_kg=100.0, com_local=(1.0, 0.0))
    assert link.inertia_kgm2 > 0


def test_units():
    assert abs(bar_to_Pa(1.0) - 1e5) < 1e-9
    assert abs(LPM_to_m3s(60000.0) - 1.0) < 1e-9
