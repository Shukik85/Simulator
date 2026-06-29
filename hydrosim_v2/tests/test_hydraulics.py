import math

import numpy as np
import pytest

from hydrosim_v2.config.hydraulics import (
    LSPumpConfig, LSValveSectionConfig, ReliefValveConfig,
    CylinderDynamicsConfig, FluidConfig,
)
from hydrosim_v2.config.base import CylinderGeometry, Attachment2D
from hydrosim_v2.hydraulics.pump import LSPumpModel
from hydrosim_v2.hydraulics.valve import LSValveSection
from hydrosim_v2.hydraulics.cylinder import HydraulicCylinder
from hydrosim_v2.hydraulics.relief import ReliefValve
from hydrosim_v2.hydraulics.integrator import rk4_step, integrate
from hydrosim_v2.core.units import bar_to_Pa


class TestLSPump:
    def test_max_flow_positive(self):
        pump = LSPumpModel(LSPumpConfig())
        assert pump.max_theoretical_flow_m3s > 0

    def test_target_displacement_standby(self):
        pump = LSPumpModel(LSPumpConfig())
        eps = pump.target_displacement(
            p_pump=bar_to_Pa(20.0),
            p_ls_max=bar_to_Pa(0.0),
        )
        assert eps > 0

    def test_target_displacement_cutoff(self):
        pump = LSPumpModel(LSPumpConfig())
        eps = pump.target_displacement(
            p_pump=bar_to_Pa(400.0),
            p_ls_max=bar_to_Pa(100.0),
        )
        assert eps == 0.0

    def test_swash_plate_dynamics(self):
        pump = LSPumpModel(LSPumpConfig())
        deps = pump.swash_plate_rhs(0.5, bar_to_Pa(200.0), bar_to_Pa(150.0))
        assert np.isfinite(deps)


class TestValveSection:
    @pytest.fixture
    def valve(self):
        return LSValveSection(LSValveSectionConfig(), FluidConfig())

    def test_spool_area_zero_at_deadband(self, valve):
        assert valve.spool_area_fraction(0.01) == 0.0

    def test_spool_area_full(self, valve):
        assert valve.spool_area_fraction(1.0) > 0

    def test_flow_pa_positive(self, valve):
        Q = valve.flow_pa(0.5, bar_to_Pa(200.0), bar_to_Pa(50.0))
        assert Q > 0

    def test_flow_pa_zero_when_reversing(self, valve):
        Q = valve.flow_pa(-0.5, bar_to_Pa(200.0), bar_to_Pa(50.0))
        assert Q == 0.0

    def test_flow_bt(self, valve):
        Q = valve.flow_bt(0.5, bar_to_Pa(50.0))
        assert Q > 0

    def test_supply_flow_positive(self, valve):
        Q = valve.supply_flow(0.5, bar_to_Pa(200.0), bar_to_Pa(50.0), bar_to_Pa(20.0))
        assert Q > 0


class TestHydraulicCylinder:
    @pytest.fixture
    def cyl(self):
        geom = CylinderGeometry(
            name="test", stroke_m=1.0, length_min_m=1.5,
            bore_diameter_m=0.110, rod_diameter_m=0.080,
            base_mount=Attachment2D(body="base", point_local=(0.0, 0.0)),
            rod_mount=Attachment2D(body="boom", point_local=(1.0, 0.0)),
        )
        dyn = CylinderDynamicsConfig()
        fluid = FluidConfig()
        return HydraulicCylinder(geom, dyn, fluid)

    def test_force_positive(self, cyl):
        F = cyl.force(bar_to_Pa(100.0), bar_to_Pa(10.0))
        assert F > 0

    def test_force_negative(self, cyl):
        F = cyl.force(bar_to_Pa(10.0), bar_to_Pa(100.0))
        assert F < 0

    def test_chamber_volumes_positive(self, cyl):
        VA, VB = cyl.chamber_volumes(0.5)
        assert VA > 0
        assert VB > 0

    def test_pressure_rhs_zero_flow(self, cyl):
        dpA, dpB = cyl.rhs(
            bar_to_Pa(50.0), bar_to_Pa(50.0), 0.5, 0.0, 0.0, 0.0,
        )
        assert abs(dpA) < 1e-6
        assert abs(dpB) < 1e-6

    def test_friction_positive(self, cyl):
        F = cyl.friction_force(0.1)
        assert F > 0

    def test_friction_negative(self, cyl):
        F = cyl.friction_force(-0.1)
        assert F < 0


class TestReliefValve:
    def test_no_flow_below_crack(self):
        rv = ReliefValve(ReliefValveConfig())
        Q = rv.flow_m3s(bar_to_Pa(200.0))
        assert Q == 0.0

    def test_flow_above_crack(self):
        rv = ReliefValve(ReliefValveConfig())
        Q = rv.flow_m3s(bar_to_Pa(350.0))
        assert Q > 0


class TestIntegrator:
    def test_rk4_on_linear_ode(self):
        def f(t, s):
            return -s  # ds/dt = -s ⇒ s(t) = s0 * exp(-t)
        s0 = np.array([1.0])
        ts, ss = integrate(f, s0, (0.0, 1.0), 0.01)
        expected = np.exp(-ts)
        assert np.allclose(ss.flatten(), expected, atol=1e-4)
