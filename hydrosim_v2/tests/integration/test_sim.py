"""Integration smoke test for the full simulation loop."""

import numpy as np

from hydrosim_v2.config import ExcavatorConfig, DEFAULT_MECHANICS_CONFIG, LSConfig
from hydrosim_v2.core.sim_state import SimState
from hydrosim_v2.hydraulics import LSPump, LSValveSection, Cylinder, ReliefValve, SimRHS
from hydrosim_v2.mechanics.loads import LoadModel

CYL_NAMES = ("boom_cyl", "arm_cyl", "bucket_cyl")
SEC_NAMES = ("boom", "arm", "bucket")


def _make_sim():
    cfg = ExcavatorConfig(mechanics=DEFAULT_MECHANICS_CONFIG, hydraulics=LSConfig())
    mech = cfg.mechanics
    ls_cfg = cfg.hydraulics

    pump = LSPump(ls_cfg.pump)
    relief = ReliefValve("main", ls_cfg.relief)
    valves = {}
    cylinders = {}
    for cn, sn in zip(CYL_NAMES, SEC_NAMES):
        geo = mech.cylinders()[cn]
        valves[cn] = LSValveSection(cn, ls_cfg.valve_sections[sn])
        cylinders[cn] = Cylinder(cn, geo, ls_cfg.cylinder_dynamics[sn], ls_cfg.fluid)

    loads = LoadModel(cfg)
    rhs = SimRHS(cfg, pump, valves, cylinders, relief, loads)
    return rhs, cfg


def test_sim_rhs_creates_and_steps():
    rhs, _ = _make_sim()

    state = SimState()
    state.cyl_pos = {"boom_cyl": 0.5, "arm_cyl": 0.5, "bucket_cyl": 0.4}
    state.p_a = {"boom_cyl": 50e5, "arm_cyl": 30e5, "bucket_cyl": 20e5}
    state.p_b = {"boom_cyl": 10e5, "arm_cyl": 10e5, "bucket_cyl": 10e5}

    spools = {"boom_cyl": 0.3, "arm_cyl": -0.2, "bucket_cyl": 0.1}

    ds = rhs.derivatives(state, spools, 1800.0)
    assert ds is not None
    for name in CYL_NAMES:
        assert np.isfinite(ds.p_a[name])
        assert np.isfinite(ds.p_b[name])
        assert np.isfinite(ds.cyl_pos[name])
        assert np.isfinite(ds.cyl_vel[name])

    ns = rhs.euler_step(state, spools, 1800.0, 0.001)
    assert ns.t == 0.001
    for name in CYL_NAMES:
        assert np.isfinite(ns.cyl_pos[name])
        assert np.isfinite(ns.p_a[name])
        assert ns.cyl_pos[name] >= 0.0

    ns4 = rhs.rk4_step(state, spools, 1800.0, 0.001)
    assert ns4.t == 0.001
    for name in CYL_NAMES:
        assert np.isfinite(ns4.cyl_pos[name])


def test_sim_rhs_multi_step():
    rhs, _ = _make_sim()

    state = SimState()
    state.cyl_pos = {"boom_cyl": 0.5, "arm_cyl": 0.5, "bucket_cyl": 0.4}

    spools = {"boom_cyl": 0.0, "arm_cyl": 0.0, "bucket_cyl": 0.0}
    for _ in range(10):
        state = rhs.euler_step(state, spools, 1500.0, 0.001)

    assert abs(state.t - 0.01) < 1e-12
    for name in CYL_NAMES:
        assert np.isfinite(state.cyl_pos[name])


def test_sim_rhs_zero_spool():
    rhs, _ = _make_sim()

    state = SimState()
    spools = {"boom_cyl": 0.0, "arm_cyl": 0.0, "bucket_cyl": 0.0}
    ns = rhs.rk4_step(state, spools, 1200.0, 0.002)
    assert ns.t == 0.002
    assert np.isfinite(ns.p_pump)
    assert ns.p_pump > 0.0
