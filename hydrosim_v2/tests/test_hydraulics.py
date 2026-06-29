import pytest

from hydrosim_v2.config import DEFAULT_MECHANICS_CONFIG, LSConfig
from hydrosim_v2.config.base import CylinderGeometry, Attachment2D
from hydrosim_v2.config.hydraulics import (
    LSPumpConfig, LSValveSectionConfig, ReliefValveConfig,
    CylinderDynamicsConfig, FluidConfig,
)
from hydrosim_v2.hydraulics import LSPump, LSValveSection, Cylinder, ReliefValve


@pytest.fixture
def pump_cfg() -> LSPumpConfig:
    return LSPumpConfig()


@pytest.fixture
def pump(pump_cfg) -> LSPump:
    return LSPump(pump_cfg)


@pytest.fixture
def valve_cfg() -> LSValveSectionConfig:
    return LSValveSectionConfig()


@pytest.fixture
def valve(valve_cfg) -> LSValveSection:
    return LSValveSection("test", valve_cfg)


@pytest.fixture
def cylinder_geo() -> CylinderGeometry:
    return CylinderGeometry(
        name="test_cyl",
        stroke_m=1.0,
        length_min_m=1.5,
        bore_diameter_m=0.110,
        rod_diameter_m=0.080,
        base_mount=Attachment2D(body="base", point_local=(0.0, 0.0)),
        rod_mount=Attachment2D(body="boom", point_local=(1.0, 1.0)),
    )


@pytest.fixture
def cylinder_dyn() -> CylinderDynamicsConfig:
    return CylinderDynamicsConfig()


@pytest.fixture
def fluid() -> FluidConfig:
    return FluidConfig()


@pytest.fixture
def cylinder(cylinder_geo, cylinder_dyn, fluid) -> Cylinder:
    return Cylinder("test_cyl", cylinder_geo, cylinder_dyn, fluid)


class TestLSPump:
    def test_max_flow_positive(self, pump):
        assert pump.max_flow_m3_s > 0

    def test_initial_state(self, pump):
        assert pump.swash == 0.0
        assert pump.q_out == 0.0

    def test_step_returns_flow(self, pump):
        q = pump.step(2200.0, 50e5, 30e5, 0.01)
        assert q >= 0


class TestLSValveSection:
    def test_initial_state(self, valve):
        assert valve.spool == 0.0
        assert valve.q_p == 0.0
        assert valve.q_t == 0.0

    def test_positive_spool_opens_p_to_a(self, valve):
        valve.step(0.5, 200e5, 0.0, 0.0, 0.0)
        assert valve.q_a > 0

    def test_negative_spool_opens_p_to_b(self, valve):
        valve.step(-0.5, 200e5, 0.0, 0.0, 0.0)
        assert valve.q_b > 0


class TestCylinder:
    def test_initial_state(self, cylinder):
        assert cylinder.position == 0.0
        assert cylinder.velocity == 0.0

    def test_length_at_min(self, cylinder):
        assert cylinder.length() == cylinder.geo.length_min_m

    def test_area_properties(self, cylinder_geo):
        assert cylinder_geo.area_piston_m2 > cylinder_geo.area_annulus_m2
        assert cylinder_geo.area_annulus_m2 > 0


class TestReliefValve:
    def test_below_crack_no_flow(self):
        relief = ReliefValve("test", ReliefValveConfig(crack_bar=300.0))
        q = relief.flow(200e5)
        assert q == 0.0

    def test_above_crack_has_flow(self):
        relief = ReliefValve("test", ReliefValveConfig(crack_bar=300.0))
        q = relief.flow(350e5)
        assert q > 0
